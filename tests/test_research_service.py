import copy
import inspect

import pytest

import nexora.research.models as models_module
import nexora.research.service as service_module
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.prompt import ResearchPromptBuilder
from nexora.research.models import ResearchAnswer
from nexora.research.service import ResearchAnswerService
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem


class FakeLLMService:
    def __init__(self, response=None, error=None):
        self.response = response or LLMResponse(
            text="Grounded answer [E1].", model="served-model",
            usage={"input_tokens": 10, "output_tokens": 5}, raw={"response_id": "r1"},
        )
        self.error = error
        self.calls = []

    def generate(self, messages, *, model=None, temperature=None, max_output_tokens=None):
        self.calls.append((messages, model, temperature, max_output_tokens))
        if self.error:
            raise self.error
        return self.response


class RecordingBuilder(ResearchPromptBuilder):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.messages = [LLMMessage(role="system", content="s"), LLMMessage(role="user", content="u")]

    def build(self, query, context):
        self.calls.append((query, context))
        return self.messages


def make_context(n=2):
    items = [
        EvidenceItem(document_id=f"d{i}", chunk_id=f"c{i}", text=f"text {i}", score=0.5,
                     metadata={"i": i}, position=i)
        for i in range(n)
    ]
    return EvidenceContext(query="q", items=items, total_candidates=n)


def test_valid_answer_flow_and_mapping():
    llm = FakeLLMService()
    result = ResearchAnswerService(llm).answer("What is X?", make_context(2))
    assert isinstance(result, ResearchAnswer)
    assert result.text == "Grounded answer [E1]." and result.model == "served-model"
    assert result.query == "What is X?"
    assert result.usage == {"input_tokens": 10, "output_tokens": 5}
    assert result.raw == {"response_id": "r1"}
    assert result.evidence_count == 2
    assert len(llm.calls) == 1


def test_builder_and_llm_receive_expected_values():
    llm, builder, ctx = FakeLLMService(), RecordingBuilder(), make_context()
    ResearchAnswerService(llm, prompt_builder=builder).answer("my query", ctx)
    assert len(builder.calls) == 1
    assert builder.calls[0][0] == "my query" and builder.calls[0][1] is ctx
    assert llm.calls[0][0] is builder.messages


def test_default_prompt_builder_produces_system_and_user_messages():
    llm = FakeLLMService()
    ResearchAnswerService(llm).answer("q", make_context(1))
    messages = llm.calls[0][0]
    assert [m.role for m in messages] == ["system", "user"]
    assert "text 0" in messages[1].content


def test_options_forwarded_unchanged():
    llm = FakeLLMService()
    ResearchAnswerService(llm).answer("q", make_context(), model="m2", temperature=0.4, max_output_tokens=77)
    assert llm.calls[0][1:] == ("m2", 0.4, 77)
    ResearchAnswerService(llm).answer("q", make_context())
    assert llm.calls[1][1:] == (None, None, None)


def test_empty_evidence_count_zero():
    assert ResearchAnswerService(FakeLLMService()).answer("q", make_context(0)).evidence_count == 0


def test_none_usage_and_raw_pass_through():
    llm = FakeLLMService(response=LLMResponse(text="t", model="m"))
    result = ResearchAnswerService(llm).answer("q", make_context())
    assert result.usage is None and result.raw is None


@pytest.mark.parametrize("query,exc", [("", ValueError), ("  \n", ValueError), (None, TypeError), (5, TypeError)])
def test_invalid_query_rejected_before_llm(query, exc):
    llm = FakeLLMService()
    with pytest.raises(exc):
        ResearchAnswerService(llm).answer(query, make_context())
    assert llm.calls == []


@pytest.mark.parametrize("ctx", [None, "evidence", [], {"items": []}])
def test_invalid_context_rejected_before_llm(ctx):
    llm = FakeLLMService()
    with pytest.raises(TypeError):
        ResearchAnswerService(llm).answer("q", ctx)
    assert llm.calls == []


def test_constructor_validation_and_no_call_on_construction():
    llm = FakeLLMService()
    ResearchAnswerService(llm)
    assert llm.calls == []
    with pytest.raises(TypeError):
        ResearchAnswerService(object())
    with pytest.raises(TypeError):
        ResearchAnswerService(llm, prompt_builder="builder")


def test_llm_errors_propagate_unchanged():
    err = RuntimeError("provider down")
    with pytest.raises(RuntimeError) as info:
        ResearchAnswerService(FakeLLMService(error=err)).answer("q", make_context())
    assert info.value is err


def test_context_not_mutated():
    ctx = make_context(3)
    before = copy.deepcopy(ctx)
    ResearchAnswerService(FakeLLMService()).answer("q", ctx)
    assert ctx == before


def test_deterministic_repeated_calls():
    service = ResearchAnswerService(FakeLLMService())
    ctx = make_context()
    assert service.answer("q", ctx) == service.answer("q", ctx)


def test_no_openai_vector_store_or_embedding_dependencies():
    for module in (service_module, models_module):
        source = inspect.getsource(module).lower()
        for forbidden in ("openai", "qdrant", "vectorstore", "embedding", "os.environ"):
            assert forbidden not in source
