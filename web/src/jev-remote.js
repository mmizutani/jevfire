import { UNIT_DEFINITIONS } from './contract.js';
import { decisionChoices } from './decision-prompt.js';
import {
  DEFAULT_PROMPTS,
  CHARACTER_PROMPTS,
  ACTION_DESCRIPTIONS,
} from './prompts.js';
import { describeObservation } from './observation.js';
import {
  DRIVERS,
  driverChoices,
  ACTION_LABELS as DRIVING_LABELS,
} from './driving/contract.js';
import { CONTROL_SCHEMA, DEFAULT_POLICY } from './mario/contract.js';
import { buildMarioPromptParts } from './mario/prompt.js';
import {
  buildMarioManeuverPrompt,
  DEFAULT_MANEUVER_POLICY,
} from './mario/maneuver-prompt.js';

const model = 'diffusiongemma';
const choice = (instructions, criteria) => ({
  type: 'choice',
  instructions,
  criteria,
});
const criteriaFor = (options, descriptions) =>
  Object.fromEntries(
    options.map((option) => [
      String(option),
      descriptions[String(option)] ?? String(option),
    ]),
  );
const requestFor = (state, questions) => ({
  model,
  state,
  questions,
  samples: 1,
  steps: 1,
});

export function buildJevRequest(message) {
  if (!message || typeof message !== 'object')
    throw new Error('Invalid Jev request');
  if (message.type === 'decide' && message.scenario !== 'driving') {
    const unit = UNIT_DEFINITIONS.find((item) => item.id === message.unitId);
    if (!unit) throw new Error('Unknown villager');
    const actions = decisionChoices(unit.id, message.actions);
    const mission = message.mission || 'Keep the villagers and hearth alive.';
    const policy = message.rolePrompt ?? DEFAULT_PROMPTS[unit.role];
    if (
      typeof mission !== 'string' ||
      mission.length > 500 ||
      typeof policy !== 'string' ||
      !policy.trim() ||
      policy.length > 1500
    )
      throw new Error('Invalid village instructions');
    const state = `Village order: ${mission}\nSelected villager: ${unit.name} (${unit.id}).\nCurrent observations:\n${describeObservation(message.context)}`;
    const questions = {
      [unit.id]: choice(
        `Choose ${unit.name}'s next ${unit.role} job to help the village survive. Personality: ${unit.personality}. Role policy: ${policy} Character priorities: ${CHARACTER_PROMPTS[unit.id]}`,
        criteriaFor(actions, ACTION_DESCRIPTIONS),
      ),
    };
    return {
      request: requestFor(state, questions),
      options: { [unit.id]: actions },
    };
  }
  if (
    message.type === 'decideDrivingBatch' ||
    (message.type === 'decide' && message.scenario === 'driving')
  ) {
    const rows =
      message.type === 'decideDrivingBatch' ? message.requests : [message];
    if (!Array.isArray(rows) || !rows.length || rows.length > DRIVERS.length)
      throw new Error('Invalid driver requests');
    const questions = {},
      options = {},
      states = [];
    for (const row of rows) {
      const driver = DRIVERS.find((item) => item.id === row.unitId);
      if (
        !driver ||
        Object.hasOwn(questions, row.unitId) ||
        row.context?.self?.id !== row.unitId
      )
        throw new Error('Invalid driver identity');
      const actions = driverChoices(
        row.unitId,
        row.actions ?? row.context?.availableActions,
      );
      const policy = row.rolePrompt ?? driver.defaultPrompt;
      if (typeof policy !== 'string' || !policy.trim() || policy.length > 1000)
        throw new Error('Invalid driver policy');
      questions[row.unitId] = choice(
        `Choose ${driver.name}'s next racing action. Follow this driver's strategy: ${policy}. Brake for unsafe corners or traffic; use only the currently available maneuvers.`,
        criteriaFor(actions, DRIVING_LABELS),
      );
      options[row.unitId] = actions;
      states.push({ driver: row.unitId, observation: row.context });
    }
    const state = JSON.stringify({
      raceOrder:
        message.mission || 'Finish three laps ahead of the other cars.',
      drivers: states,
    });
    if (state.length > 65000)
      throw new Error('Driving observation is too large');
    return { request: requestFor(state, questions), options };
  }
  if (message.type === 'decideMario') {
    const policy = message.rolePrompt ?? DEFAULT_POLICY;
    buildMarioPromptParts(message.context, message.mission ?? '', policy);
    const options = Object.fromEntries(
      Object.entries(CONTROL_SCHEMA).map(([key, values]) => [key, [...values]]),
    );
    const questions = {
      direction: choice(
        `Select horizontal movement to reach the flag alive. Policy: ${policy}`,
        {
          left: 'Move left',
          still: 'Do not move horizontally',
          right: 'Move right toward the flag',
        },
      ),
      jump: choice(
        `Select jump button state. A new press while grounded starts a jump; holding extends it. Policy: ${policy}`,
        {
          false: 'Release jump',
          true: 'Press or hold jump',
        },
      ),
      speed: choice(`Select movement speed. Policy: ${policy}`, {
        walk: 'Walk, up to 4.5 tiles per second',
        run: 'Run, up to 8.4 tiles per second',
      }),
    };
    return {
      request: requestFor(JSON.stringify(message.context), questions),
      options,
    };
  }
  if (message.type === 'decideMarioManeuver') {
    const policy = message.rolePrompt ?? DEFAULT_MANEUVER_POLICY;
    buildMarioManeuverPrompt(message.context, policy);
    const maneuvers = message.context.options;
    const options = { maneuver: maneuvers.map((item) => item.id) };
    const questions = {
      maneuver: choice(
        `Choose a safe maneuver toward the flag. The local physics predictor's forward gains are approximate. Policy: ${policy}`,
        Object.fromEntries(
          maneuvers.map((item) => [
            item.id,
            `Predicted forward gain ${Number(item.progress.toFixed(1))} tiles`,
          ]),
        ),
      ),
    };
    return {
      request: requestFor(JSON.stringify({ options: maneuvers }), questions),
      options,
    };
  }
  throw new Error('Unknown Jev operation');
}

export function parseJevResponse(body, options) {
  if (
    !body ||
    typeof body !== 'object' ||
    !body.answers ||
    typeof body.answers !== 'object' ||
    !Number.isInteger(body.usage?.input_tokens) ||
    body.usage.input_tokens < 0 ||
    !Number.isInteger(body.usage?.output_tokens) ||
    body.usage.output_tokens < 0
  )
    throw new Error('Invalid Jev response');
  const parsed_json = {},
    fields = {};
  for (const [id, choices] of Object.entries(options)) {
    const answer = body.answers[id];
    if (
      answer?.type !== 'choice' ||
      !answer.probabilities ||
      typeof answer.probabilities !== 'object'
    )
      throw new Error(`Missing Jev choice for ${id}`);
    const value = choices.find((item) => String(item) === answer.choice);
    if (value === undefined) throw new Error(`Invalid Jev choice for ${id}`);
    if (
      Object.keys(answer.probabilities).length !== choices.length ||
      choices.some((item) => !Object.hasOwn(answer.probabilities, String(item)))
    )
      throw new Error(`Invalid Jev probability labels for ${id}`);
    const probabilities = choices.map(
      (item) => answer.probabilities[String(item)],
    );
    if (probabilities.some((p) => !Number.isFinite(p) || p < 0 || p > 1))
      throw new Error(`Invalid Jev probabilities for ${id}`);
    if (Math.abs(probabilities.reduce((sum, p) => sum + p, 0) - 1) > 0.001)
      throw new Error(`Jev probabilities do not sum to one for ${id}`);
    if (answer.probabilities[String(value)] < Math.max(...probabilities) - 1e-9)
      throw new Error(`Jev choice contradicts probabilities for ${id}`);
    parsed_json[id] = value;
    fields[id] = { value, options: choices, probabilities };
  }
  return {
    parsed_json,
    fields,
    fields_scored: Object.keys(options).length,
    scores_are_calibrated: false,
    usage: {
      input_tokens: body.usage.input_tokens,
      output_tokens: body.usage.output_tokens,
      cached_prefix_tokens: 0,
    },
  };
}

export async function callJev(request, options, fetcher = fetch) {
  let response;
  try {
    response = await fetcher('/v1/systemone', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(request),
    });
  } catch (error) {
    throw new Error(
      `Jev server unreachable: ${error.message || String(error)}`,
    );
  }
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Jev server returned invalid JSON (${response.status})`);
  }
  if (!response.ok)
    throw new Error(
      `Jev server ${response.status}: ${body?.error?.message || 'Request failed'}`,
    );
  return { ...parseJevResponse(body, options), model: body.model ?? model };
}
