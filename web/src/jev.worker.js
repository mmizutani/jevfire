import { buildJevRequest, callJev } from './jev-remote.js';

const send = (type, value = {}) => self.postMessage({ type, ...value });
let ready = false;
let tail = Promise.resolve();

async function load() {
  let response;
  try {
    response = await fetch('/jev-health');
  } catch (error) {
    throw new Error(
      `Jev server unreachable: ${error.message || String(error)}`,
    );
  }
  if (!response.ok)
    throw new Error(`Jev server health check failed (${response.status})`);
  const health = await response.json();
  if (health.status !== 'ok') throw new Error('Jev server is not ready');
  ready = true;
  send('ready', {
    model: 'diffusiongemma',
    sdk_backend: 'remote-jev',
    execution: 'one-server-request',
  });
}

function decision(message, result, elapsed, usage = result.usage) {
  return {
    ...result,
    type: 'decision',
    id: message.id,
    epoch: message.epoch,
    unitId: message.unitId,
    scenario: message.scenario,
    model: result.model,
    elapsed_ms: elapsed,
    prompt_tokens: usage.input_tokens,
    usage,
    execution: 'remote-jev',
  };
}

async function handle(message) {
  if (message.type === 'load') return load();
  if (!ready) throw new Error('Connect to the Jev server first');
  const { request, options } = buildJevRequest(message);
  const start = performance.now();
  if (message.type === 'decideDrivingBatch') {
    const result = await callJev(request, options);
    const elapsed = performance.now() - start;
    const decisions = message.requests.map((row, index) => {
      const unitId = row.unitId;
      const item = decision(
        { ...message, unitId },
        {
          parsed_json: { [unitId]: result.parsed_json[unitId] },
          fields: { [unitId]: result.fields[unitId] },
          fields_scored: 1,
          scores_are_calibrated: false,
          model: result.model,
        },
        elapsed,
        { input_tokens: 0, output_tokens: 0, cached_prefix_tokens: 0 },
      );
      send('drivingField', {
        id: message.id,
        epoch: message.epoch,
        scenario: 'driving',
        unitId,
        decision: item,
        batch_size: message.requests.length,
        index,
        fleet_age_ms: elapsed,
      });
      return item;
    });
    send('drivingBatch', {
      id: message.id,
      epoch: message.epoch,
      scenario: 'driving',
      decisions,
      parsed_json: result.parsed_json,
      fields: result.fields,
      fields_scored: decisions.length,
      scores_are_calibrated: false,
      elapsed_ms: elapsed,
      usage: result.usage,
      execution: 'one-shared-jev-request',
    });
    return;
  }
  const result = await callJev(request, options);
  const elapsed = performance.now() - start;
  if (
    message.type === 'decideMario' ||
    message.type === 'decideMarioManeuver'
  ) {
    send(message.type === 'decideMario' ? 'marioDecision' : 'marioManeuver', {
      ...result,
      id: message.id,
      epoch: message.epoch,
      scenario: 'mario',
      elapsed_ms: elapsed,
      execution: 'remote-jev',
    });
    return;
  }
  send('decision', decision(message, result, elapsed));
}

if (typeof self !== 'undefined')
  self.onmessage = ({ data }) => {
    const snapshot = structuredClone(data);
    tail = tail
      .then(() => handle(snapshot))
      .catch((error) => {
        send('error', {
          message: error.message || String(error),
          id: snapshot?.id,
          epoch: snapshot?.epoch,
          operation: snapshot?.type,
        });
      });
  };
