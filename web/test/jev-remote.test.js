import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildJevRequest,
  parseJevResponse,
  callJev,
} from '../src/jev-remote.js';
import { Game } from '../src/game.js';
import { DrivingGame } from '../src/driving/game.js';
import { MarioGame } from '../src/mario/game.js';

test('village request carries current observation, policy, and legal action descriptions', () => {
  const game = new Game({ seed: 7 });
  const unit = game.units[0];
  const context = game.contextFor(unit.id);
  const actions = game.availableActions(unit);
  const { request, options } = buildJevRequest({
    type: 'decide',
    unitId: unit.id,
    context,
    actions,
    mission: 'Protect the hearth.',
    rolePrompt: 'Gather safely.',
  });
  assert.equal(request.model, 'diffusiongemma');
  assert.equal(request.samples, 1);
  assert.equal(request.steps, 1);
  assert.match(request.state, /Protect the hearth/);
  assert.match(request.state, /Food stored at home/);
  assert.match(request.questions[unit.id].instructions, /Gather safely/);
  assert.deepEqual(Object.keys(request.questions[unit.id].criteria), actions);
  assert.deepEqual(options[unit.id], actions);
});

test('driving fleet is one request with distinct policies and current legal actions', () => {
  const game = new DrivingGame();
  const requests = game.drivers.map((car) => ({
    unitId: car.id,
    context: game.contextFor(car.id),
    actions: game.contextFor(car.id).availableActions,
    rolePrompt: `Policy for ${car.id}`,
  }));
  const { request } = buildJevRequest({
    type: 'decideDrivingBatch',
    scenario: 'driving',
    mission: 'Finish first.',
    requests,
  });
  assert.equal(Object.keys(request.questions).length, 4);
  assert.equal(request.model, 'diffusiongemma');
  for (const row of requests) {
    assert.match(
      request.questions[row.unitId].instructions,
      new RegExp(row.rolePrompt),
    );
    assert.deepEqual(
      Object.keys(request.questions[row.unitId].criteria),
      row.actions,
    );
    assert.match(request.state, new RegExp(row.unitId));
  }
});

test('Mario raw fields preserve booleans and maneuver IDs', () => {
  const game = new MarioGame();
  const raw = buildJevRequest({
    type: 'decideMario',
    context: game.observe(),
    rolePrompt: 'Reach the flag.',
  });
  assert.deepEqual(Object.keys(raw.request.questions), [
    'direction',
    'jump',
    'speed',
  ]);
  const response = {
    answers: {
      direction: {
        type: 'choice',
        choice: 'right',
        probabilities: { left: 0.1, still: 0.1, right: 0.8 },
      },
      jump: {
        type: 'choice',
        choice: 'true',
        probabilities: { false: 0.2, true: 0.8 },
      },
      speed: {
        type: 'choice',
        choice: 'run',
        probabilities: { walk: 0.2, run: 0.8 },
      },
    },
    usage: { input_tokens: 100, output_tokens: 3 },
  };
  const result = parseJevResponse(response, raw.options);
  assert.deepEqual(result.parsed_json, {
    direction: 'right',
    jump: true,
    speed: 'run',
  });
  assert.equal(result.usage.input_tokens, 100);
  assert.equal(result.usage.cached_prefix_tokens, 0);
  assert.deepEqual(result.fields.jump.options, [false, true]);
  const maneuvers = buildJevRequest({
    type: 'decideMarioManeuver',
    context: {
      options: [
        { id: 'jump_right', progress: 3 },
        { id: 'run_right', progress: 4 },
      ],
    },
    rolePrompt: 'Reach the flag.',
  });
  assert.deepEqual(Object.keys(maneuvers.request.questions.maneuver.criteria), [
    'jump_right',
    'run_right',
  ]);
});

test('malformed and incomplete Jev answers fail closed', () => {
  const options = { action: ['accelerate', 'brake'] };
  const valid = {
    answers: {
      action: {
        type: 'choice',
        choice: 'brake',
        probabilities: { accelerate: 0.3, brake: 0.7 },
      },
    },
    usage: { input_tokens: 10, output_tokens: 1 },
  };
  assert.equal(parseJevResponse(valid, options).parsed_json.action, 'brake');
  assert.throws(() => parseJevResponse({ ...valid, answers: {} }, options));
  assert.throws(() =>
    parseJevResponse(
      {
        ...valid,
        answers: { action: { ...valid.answers.action, choice: 'pit' } },
      },
      options,
    ),
  );
  assert.throws(() =>
    parseJevResponse(
      {
        ...valid,
        answers: {
          action: {
            ...valid.answers.action,
            probabilities: { accelerate: NaN, brake: 0.7 },
          },
        },
      },
      options,
    ),
  );
  assert.throws(() => parseJevResponse({ ...valid, usage: {} }, options));
  for (const probabilities of [
    { accelerate: 0.3, brake: 0.7, pit: 0 },
    { accelerate: 0.1, brake: 0.2 },
    { accelerate: 0.8, brake: 0.2 },
  ]) {
    assert.throws(() =>
      parseJevResponse(
        {
          ...valid,
          answers: { action: { ...valid.answers.action, probabilities } },
        },
        options,
      ),
    );
  }
});

test('HTTP adapter posts typed request and reports upstream failures', async () => {
  const request = { model: 'diffusiongemma', state: 'race', questions: {} };
  const options = { action: ['accelerate', 'brake'] };
  const body = {
    answers: {
      action: {
        type: 'choice',
        choice: 'brake',
        probabilities: { accelerate: 0.3, brake: 0.7 },
      },
    },
    usage: { input_tokens: 10, output_tokens: 1 },
  };
  const result = await callJev(request, options, async (path, init) => {
    assert.equal(path, '/v1/systemone');
    assert.equal(init.method, 'POST');
    assert.equal(init.headers['content-type'], 'application/json');
    assert.deepEqual(JSON.parse(init.body), request);
    return { ok: true, json: async () => body };
  });
  assert.equal(result.parsed_json.action, 'brake');
  await assert.rejects(
    callJev(request, options, async () => ({
      ok: false,
      status: 422,
      json: async () => ({ error: { message: 'bad choice' } }),
    })),
    /Jev server 422: bad choice/,
  );
  await assert.rejects(
    callJev(request, options, async () => {
      throw new TypeError('connection refused');
    }),
    /Jev server unreachable: connection refused/,
  );
});
