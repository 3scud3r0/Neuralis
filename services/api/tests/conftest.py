import pytest

from neuralis.storage.repository import Repository
from neuralis.runtime.supervisor import Supervisor


@pytest.fixture
def supervisor(tmp_path):
    value = Supervisor(Repository(tmp_path))
    yield value
    value.shutdown()
