"""Attribute keys used on exported spans.

Standard keys come from the OpenTelemetry GenAI semantic conventions, which
are still at Development status upstream, so they are isolated here and can be
renamed in one place if the conventions change.

Custom keys live under the ``agent_state.`` prefix so they never collide with
standard attributes.
"""

# --- OpenTelemetry GenAI semantic conventions (Development status) ---
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

OP_INVOKE_AGENT = "invoke_agent"
OP_EXECUTE_TOOL = "execute_tool"

# --- Custom attributes (root span) ---
PREFIX = "agent_state."
RUN_KEY = PREFIX + "run_key"
INPUT_FP = PREFIX + "input_fp"
PATH = PREFIX + "path"
PATH_SIG = PREFIX + "path_sig"
STEPS = PREFIX + "steps"
TERMINAL = PREFIX + "terminal"
OUTCOME = PREFIX + "outcome"
SAMPLE_HINT = PREFIX + "sample_hint"
PATH_SEEN = PREFIX + "path_seen"
REF_PATH = PREFIX + "ref_path"
REF_TRACE_ID = PREFIX + "ref_trace_id"
FORK_STEP = PREFIX + "fork_step"
FORK_FROM = PREFIX + "fork_from"
FORK_TO = PREFIX + "fork_to"
FORK_TO_REF = PREFIX + "fork_to_ref"
FORK_CAUSE = PREFIX + "fork_cause"
FORK_CAUSE_STEP = PREFIX + "fork_cause_step"

# --- Custom attributes (state span) ---
STEP = PREFIX + "step"
FROM = PREFIX + "from"
TO = PREFIX + "to"
REASON = PREFIX + "reason"
DECIDED_BY = PREFIX + "decided_by"
EVIDENCE_PREFIX = PREFIX + "ev."

STATE_SPAN_PREFIX = "agent_state "
START_STATE = "START"

OUTCOME_OK = "ok"
OUTCOME_ERROR = "error"
OUTCOME_PARTIAL = "partial"

HINT_ERROR = "error"
HINT_DIVERGENT = "divergent"
HINT_RARE = "rare_path"
HINT_COMMON = "common"
KEEP_HINTS = (HINT_ERROR, HINT_DIVERGENT, HINT_RARE)

CAUSE_SAMPLING = "sampling"
CAUSE_UNRECORDED = "unrecorded"
