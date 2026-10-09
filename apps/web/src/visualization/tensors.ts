import { clamp, summarize, type NumericSummary } from "./numerics";

export interface DenseTensor {
  data: Float64Array;
  shape: number[];
  strides: number[];
}
export interface TensorSlice {
  matrix: number[][];
  rowAxis: number;
  columnAxis: number;
  fixedIndices: number[];
  sourceShape: number[];
}
export type TensorReduction = "mean" | "sum" | "min" | "max" | "l2";
export interface TensorCoordinate {
  index: number[];
  value: number;
  offset: number;
}

export function tensorSize(shape: readonly number[]): number {
  if (shape.length > 16)
    throw new Error("Tensor previews support at most 16 axes.");
  if (shape.some((size) => !Number.isSafeInteger(size) || size < 0))
    throw new Error("Tensor dimensions must be non-negative integers.");
  const size = shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(size) || size > 2000000)
    throw new Error(
      "Tensor preview exceeds the two-million-element capture budget.",
    );
  return size;
}

export function contiguousStrides(shape: readonly number[]): number[] {
  tensorSize(shape);
  let stride = 1;
  const strides = Array<number>(shape.length);
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    strides[axis] = stride;
    stride *= shape[axis];
  }
  return strides;
}

export function denseTensor(
  values: readonly number[] | Float64Array,
  shape: readonly number[],
): DenseTensor {
  const expected = tensorSize(shape);
  if (values.length !== expected)
    throw new Error(
      `Tensor shape requires ${expected} values, received ${values.length}.`,
    );
  return {
    data: Float64Array.from(values),
    shape: [...shape],
    strides: contiguousStrides(shape),
  };
}

export function validateTensor(tensor: DenseTensor): void {
  if (tensor.data.length !== tensorSize(tensor.shape))
    throw new Error("Tensor buffer and shape disagree.");
  const expected = contiguousStrides(tensor.shape);
  if (
    tensor.strides.length !== expected.length ||
    tensor.strides.some((stride, axis) => stride !== expected[axis])
  )
    throw new Error("Tensor must use contiguous row-major strides.");
}

export function flattenIndex(
  index: readonly number[],
  shape: readonly number[],
  strides = contiguousStrides(shape),
): number {
  if (index.length !== shape.length || strides.length !== shape.length)
    throw new Error("Index rank does not match tensor rank.");
  let offset = 0;
  for (let axis = 0; axis < shape.length; axis++) {
    if (
      !Number.isInteger(index[axis]) ||
      index[axis] < 0 ||
      index[axis] >= shape[axis]
    )
      throw new Error(`Index on axis ${axis} is out of bounds.`);
    offset += index[axis] * strides[axis];
  }
  return offset;
}

export function unravelIndex(
  offset: number,
  shape: readonly number[],
): number[] {
  const size = tensorSize(shape);
  if (!Number.isInteger(offset) || offset < 0 || offset >= size)
    throw new Error("Flat tensor index is out of bounds.");
  const strides = contiguousStrides(shape);
  return shape.map(
    (dimension, axis) => Math.floor(offset / strides[axis]) % dimension,
  );
}

export function reshapeTensor(
  tensor: DenseTensor,
  shape: readonly number[],
): DenseTensor {
  validateTensor(tensor);
  if (tensorSize(shape) !== tensor.data.length)
    throw new Error("Reshape must preserve the number of tensor elements.");
  return {
    data: tensor.data,
    shape: [...shape],
    strides: contiguousStrides(shape),
  };
}

export function transposeTensor(
  tensor: DenseTensor,
  order: readonly number[],
): DenseTensor {
  validateTensor(tensor);
  if (
    order.length !== tensor.shape.length ||
    new Set(order).size !== order.length ||
    order.some(
      (axis) =>
        axis < 0 || axis >= tensor.shape.length || !Number.isInteger(axis),
    )
  )
    throw new Error("Transpose order must be a permutation of tensor axes.");
  const shape = order.map((axis) => tensor.shape[axis]);
  const data = new Float64Array(tensor.data.length);
  const outputStrides = contiguousStrides(shape);
  for (let offset = 0; offset < data.length; offset++) {
    let sourceOffset = 0;
    for (let axis = 0; axis < order.length; axis++) {
      const coordinate = Math.floor(offset / outputStrides[axis]) % shape[axis];
      sourceOffset += coordinate * tensor.strides[order[axis]];
    }
    data[offset] = tensor.data[sourceOffset];
  }
  return { data, shape, strides: outputStrides };
}

export function sliceTensor(
  tensor: DenseTensor,
  rowAxis: number,
  columnAxis: number,
  fixedIndices: readonly number[] = [],
): TensorSlice {
  validateTensor(tensor);
  const rank = tensor.shape.length;
  if (rank < 2)
    throw new Error(
      "A two-dimensional slice requires at least two tensor axes.",
    );
  if (
    rowAxis === columnAxis ||
    !Number.isInteger(rowAxis) ||
    !Number.isInteger(columnAxis) ||
    rowAxis < 0 ||
    columnAxis < 0 ||
    rowAxis >= rank ||
    columnAxis >= rank
  )
    throw new Error("Slice row and column axes must be distinct valid axes.");
  const fixed = tensor.shape.map((size, axis) => {
    if (axis === rowAxis || axis === columnAxis) return 0;
    const value = fixedIndices[axis] ?? 0;
    if (!Number.isInteger(value) || value < 0 || value >= size)
      throw new Error(`Fixed index on axis ${axis} is out of bounds.`);
    return value;
  });
  const origin = fixed.reduce(
    (sum, index, axis) => sum + index * tensor.strides[axis],
    0,
  );
  const matrix = Array.from({ length: tensor.shape[rowAxis] }, (_, row) =>
    Array.from(
      { length: tensor.shape[columnAxis] },
      (_, column) =>
        tensor.data[
          origin +
            row * tensor.strides[rowAxis] +
            column * tensor.strides[columnAxis]
        ],
    ),
  );
  return {
    matrix,
    rowAxis,
    columnAxis,
    fixedIndices: fixed,
    sourceShape: [...tensor.shape],
  };
}

export function reduceTensor(
  tensor: DenseTensor,
  axis: number,
  operation: TensorReduction,
): DenseTensor {
  validateTensor(tensor);
  if (!Number.isInteger(axis) || axis < 0 || axis >= tensor.shape.length)
    throw new Error("Reduction axis is out of bounds.");
  const shape = tensor.shape.filter((_, index) => index !== axis);
  const data = new Float64Array(tensorSize(shape));
  const strides = contiguousStrides(shape);
  const axisSize = tensor.shape[axis];
  for (let offset = 0; offset < data.length; offset++) {
    let sourceOffset = 0;
    let outputAxis = 0;
    for (let sourceAxis = 0; sourceAxis < tensor.shape.length; sourceAxis++) {
      if (sourceAxis === axis) continue;
      const coordinate =
        Math.floor(offset / strides[outputAxis]) % shape[outputAxis];
      sourceOffset += coordinate * tensor.strides[sourceAxis];
      outputAxis++;
    }
    let accumulated =
      operation === "min" ? Infinity : operation === "max" ? -Infinity : 0;
    for (let index = 0; index < axisSize; index++) {
      const value = tensor.data[sourceOffset + index * tensor.strides[axis]];
      if (operation === "min") accumulated = Math.min(accumulated, value);
      else if (operation === "max") accumulated = Math.max(accumulated, value);
      else accumulated += operation === "l2" ? value * value : value;
    }
    if (operation === "mean")
      accumulated = axisSize ? accumulated / axisSize : NaN;
    if (operation === "l2") accumulated = Math.sqrt(accumulated);
    data[offset] = accumulated;
  }
  return { data, shape, strides };
}

export function tensorChannelStatistics(
  tensor: DenseTensor,
  axis: number,
): NumericSummary[] {
  validateTensor(tensor);
  if (!Number.isInteger(axis) || axis < 0 || axis >= tensor.shape.length)
    throw new Error("Channel axis is out of bounds.");
  const buckets: number[][] = Array.from(
    { length: tensor.shape[axis] },
    () => [],
  );
  for (let offset = 0; offset < tensor.data.length; offset++)
    buckets[
      Math.floor(offset / tensor.strides[axis]) % tensor.shape[axis]
    ].push(tensor.data[offset]);
  return buckets.map((bucket) => summarize(bucket));
}

export function largestTensorElements(
  tensor: DenseTensor,
  count = 20,
): TensorCoordinate[] {
  validateTensor(tensor);
  const budget = clamp(Math.floor(count), 0, 512);
  const ranked: { value: number; offset: number }[] = [];
  for (let offset = 0; offset < tensor.data.length; offset++) {
    const value = tensor.data[offset];
    if (!Number.isFinite(value)) continue;
    if (
      ranked.length >= budget &&
      Math.abs(value) <= Math.abs(ranked[ranked.length - 1]?.value ?? 0)
    )
      continue;
    let insertion = 0;
    while (
      insertion < ranked.length &&
      Math.abs(ranked[insertion].value) >= Math.abs(value)
    )
      insertion++;
    ranked.splice(insertion, 0, { value, offset });
    if (ranked.length > budget) ranked.pop();
  }
  return ranked.map((element) => ({
    ...element,
    index: unravelIndex(element.offset, tensor.shape),
  }));
}

export function tensorDifference(
  left: DenseTensor,
  right: DenseTensor,
): DenseTensor {
  validateTensor(left);
  validateTensor(right);
  if (
    left.shape.length !== right.shape.length ||
    left.shape.some((size, axis) => size !== right.shape[axis])
  )
    throw new Error("Tensor difference requires identical shapes.");
  return {
    data: Float64Array.from(
      left.data,
      (value, offset) => value - right.data[offset],
    ),
    shape: [...left.shape],
    strides: [...left.strides],
  };
}

export function serializeTensor(tensor: DenseTensor): string {
  validateTensor(tensor);
  if (Array.from(tensor.data).some((value) => !Number.isFinite(value)))
    throw new Error(
      "JSON tensor export requires finite values. Use CSV to inspect invalid values.",
    );
  return JSON.stringify({
    format: "neuralis.tensor",
    version: 1,
    shape: tensor.shape,
    data: Array.from(tensor.data),
  });
}

export function parseTensor(text: string): DenseTensor {
  if (text.length > 64000000)
    throw new Error("Tensor document exceeds the import budget.");
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== "object")
    throw new Error("Expected a tensor document.");
  const record = value as Record<string, unknown>;
  if (
    record.format !== "neuralis.tensor" ||
    record.version !== 1 ||
    !Array.isArray(record.shape) ||
    !record.shape.every((size) => typeof size === "number") ||
    !Array.isArray(record.data) ||
    !record.data.every(
      (item) => typeof item === "number" && Number.isFinite(item),
    )
  )
    throw new Error("Invalid neuralis.tensor version 1 document.");
  return denseTensor(record.data as number[], record.shape as number[]);
}
