import {chromium} from 'playwright';
import {spawn} from 'node:child_process';
const root=process.cwd();
const backend='http://127.0.0.1:8001';
const id='mario-browser-check-'+Date.now();
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
let recorder;
try {
  const page=await browser.newPage({viewport:{width:1600,height:1050}});
  await page.addInitScript(() => {
    const Native=window.RTCPeerConnection;
    window.RTCPeerConnection=class extends Native {
      constructor(...args) {super(...args);window.__arenaTestPeer=this;}
    };
  });
  await page.goto('http://127.0.0.1:4173/#/research');
  await page.getByRole('button',{name:'Connect backend',exact:true}).click();
  await page.getByLabel('Extension backend URL').fill(backend);
  await page.getByRole('button',{name:'Check capabilities',exact:true}).click();
  await page.getByText('WEBRTC · available',{exact:true}).waitFor();
  await page.getByRole('tab',{name:/RL Arena/}).click();
  let log='';
  recorder=spawn('/workspace/.venvs/mario/bin/python',['arena.py','runs/smoke/latest.pt','--url',backend,'--episode-id',id,'--max-steps','120','--fps','4'],{cwd:root+'/demos/super-mario',stdio:['ignore','pipe','pipe']});
  recorder.stdout.on('data',part=>{log+=part.toString();});recorder.stderr.on('data',part=>{log+=part.toString();});
  await page.waitForTimeout(4000);
  await page.getByRole('button',{name:'Refresh episodes',exact:true}).click();
  await page.getByLabel('Live episode').selectOption(id);
  await page.getByRole('button',{name:'Connect video',exact:true}).click();
  await page.waitForTimeout(10000);
  const diagnostics=await page.evaluate(async () => {
    const peer=window.__arenaTestPeer,video=document.querySelector('.arena-live-video');
    const stats=peer?Array.from((await peer.getStats()).values()).filter(item=>['inbound-rtp','candidate-pair','transport'].includes(item.type)):[];
    return {connection:peer?.connectionState,ice:peer?.iceConnectionState,gather:peer?.iceGatheringState,
      signaling:peer?.signalingState,
      localSdp:peer?.localDescription?.sdp.split('\r\n').filter(line=>/^m=|^a=(candidate:|end-of-candidates|recvonly|sendonly|inactive|setup:|group:|mid:)/.test(line)),
      remoteSdp:peer?.remoteDescription?.sdp.split('\r\n').filter(line=>/^m=|^a=(candidate:|end-of-candidates|recvonly|sendonly|inactive|setup:|group:|mid:)/.test(line)),
      transceivers:peer?.getTransceivers().map(track=>({mid:track.mid,direction:track.direction,currentDirection:track.currentDirection,stopped:track.stopped})),
      video:{ready:video?.readyState,width:video?.videoWidth,height:video?.videoHeight},stats};
  });
  const ui=await page.locator('.arena-runtime-card').first().innerText();
  console.log('UI: '+ui.slice(-1200));
  console.log('Recorder: '+log.slice(-800));
  const candidates=diagnostics.localSdp?.filter(line=>line.startsWith('a=candidate:')) ?? [];
  if(!diagnostics.video.width && !candidates.length) {
    if(!ui.includes('no ICE candidates')) throw new Error('The browser gathered no candidates but the UI did not explain the blocked transport.');
    console.log(JSON.stringify({status:'blocked',browser_video_verified:false,
      reason:'The browser gathered zero ICE candidates. Its network or managed policy blocks local WebRTC transport.',
      diagnostics}));
  } else {
    if(!diagnostics.video.width) throw new Error('No real decoded browser video frame was received despite gathered ICE candidates.');
    console.log(JSON.stringify({status:'passed',browser_video_verified:true,diagnostics}));
  }
} finally {recorder?.kill('SIGINT');await browser.close();}
