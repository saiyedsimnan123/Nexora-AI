import copy
import inspect

import pytest

import nexora.research.query as query_module
from nexora.research.models import ResearchAnswer
from nexora.research.query import ResearchQueryService
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem


def make_context():
    item = EvidenceItem(document_id="d", chunk_id="c", text="t", score=0.5, metadata={"k": 1}, position=0)
    return EvidenceContext(query="q", items=[item], total_candidates=1)


class FakeBuilder:
    def __init__(self, error=None):
        self.error = error
        self.context = make_context()
        self.calls = []

    def build(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.context


class FakeAnswerService:
    def __init__(self, error=None):
        self.error = error
        self.answer_obj = ResearchAnswer(text="a", query="q", model="m", evidence_count=1)
        self.calls = []

    def answer(self, query, context, *, model=None, temperature=None, max_output_tokens=None):
        self.calls.append((query, context, model, temperature, max_output_tokens))
        if self.error:
            raise self.error
        return self.answer_obj


def make(**kw):
    builder, answers = FakeBuilder(kw.get("builder_error")), FakeAnswerService(kw.get("answer_error"))
    return ResearchQueryService(builder, answers), builder, answers


def test_end_to_end_orchestration():
    service, builder, answers = make()
    result = service.ask("What is X?")
    assert result is answers.answer_obj
    assert builder.calls == ["What is X?"]
    assert len(answers.calls) == 1


def test_context_and_query_passed_to_answer_service():
    service, builder, answers = make()
    service.ask("  spaced query ")
    query, context, *_ = answers.calls[0]
    assert query == "  spaced query " and context is builder.context


def test_options_forwarded_unchanged():
    service, _, answers = make()
    service.ask("q", model="m2", temperature=0.3, max_output_tokens=99)
    assert answers.calls[0][2:] == ("m2", 0.3, 99)
    service.ask("q")
    assert answers.calls[1][2:] == (None, None, None)
    service.ask("q", temperature=0)
    assert answers.calls[2][3] == 0


@pytest.mark.parametrize("query,exc", [("", ValueError), (" \n\t", ValueError), (None, TypeError), (7, TypeError), (["q"], TypeError)])
def test_invalid_query_rejected_before_any_call(query, exc):
    service, builder, answers = make()
    with pytest.raises(exc):
        service.ask(query)
    assert builder.calls == [] and answers.calls == []


def test_invalid_dependencies_rejected():
    with pytest.raises(TypeError):
        ResearchQueryService(object(), FakeAnswerService())
    with pytest.raises(TypeError):
        ResearchQueryService(FakeBuilder(), object())
    with pytest.raises(TypeError):
        ResearchQueryService(None, None)


def test_construction_makes_no_calls():
    builder, answers = FakeBuilder(), FakeAnswerService()
    ResearchQueryService(builder, answers)
    assert builder.calls == [] and answers.calls == []


@pytest.mark.parametrize("stage", ["builder_error", "answer_error"])
def test_errors_propagate_unchanged(stage):
    err = RuntimeError(stage)
    service, _, answers = make(**{stage: err})
    with pytest.raises(RuntimeError) as info:
        service.ask("q")
    assert info.value is err
    if stage == "builder_error":
        assert answers.calls == []


def test_no_mutation_of_intermediate_objects():
    service, builder, _ = make()
    before = copy.deepcopy(builder.context)
    service.ask("q")
    assert builder.context == before


def test_deterministic_repeated_orchestration():
    service, _, _ = make()
    assert service.ask("q") == service.ask("q")


def test_no_infrastructure_imports():
    source = inspect.getsource(query_module).lower()
    for forbidden in ("openai", "qdrant", "embedding", "vectorstore", "os.environ"):
        assert forbidden not in source


# ---- composition test: real quality/evidence/answer services, fake retrieval + LLM ----
from nexora.llm.models import LLMResponse  # noqa: E402
from nexora.research.service import ResearchAnswerService  # noqa: E402
from nexora.retrieval.evidence import EvidenceContextBuilder  # noqa: E402
from nexora.retrieval.models import RetrievedChunk, SearchResult  # noqa: E402
from nexora.retrieval.quality import RetrievalQualityPolicy, RetrievalQualityService  # noqa: E402


class FakeRetrieval:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query, *, limit=10):
        self.calls.append((query, limit))
        return SearchResult(query="", results=list(self.chunks))


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, messages, *, model=None, temperature=None, max_output_tokens=None):
        self.calls.append((messages, model, temperature, max_output_tokens))
        return LLMResponse(text="composed answer", model="fake-model", usage={"total_tokens": 9})


def _chunk(doc, cid, text, score):
    return RetrievedChunk(document_id=doc, chunk_id=cid, text=text, score=score, metadata={"p": 1})


def _compose(chunks, **policy_kw):
    retrieval = FakeRetrieval(chunks)
    quality = RetrievalQualityService(retrieval, RetrievalQualityPolicy(**policy_kw))
    builder = EvidenceContextBuilder(quality)  # quality service satisfies retrieve(query)
    llm = FakeLLM()
    service = ResearchQueryService(builder, ResearchAnswerService(llm))
    return service, retrieval, llm


def test_intended_composition_end_to_end():
    chunks = [
        _chunk("A", "A1", "kept evidence", 0.9),
        _chunk("A", "A1", "duplicate chunk", 0.8),
        _chunk("B", "B1", "low score evidence", 0.1),
    ]
    service, retrieval, llm = _compose(chunks, min_score=0.5, max_results=5, candidate_limit=20)
    answer = service.ask("How does X work?", model="m", temperature=0.2, max_output_tokens=50)

    assert retrieval.calls == [("How does X work?", 20)]  # limit comes from the policy
    assert answer.text == "composed answer" and answer.model == "fake-model"
    assert answer.query == "How does X work?" and answer.evidence_count == 1
    messages, model, temperature, tokens = llm.calls[0]
    assert (model, temperature, tokens) == ("m", 0.2, 50)
    user = messages[1].content
    assert "kept evidence" in user and "duplicate chunk" not in user
    assert "low score evidence" not in user


def test_composition_with_no_evidence_still_answers():
    service, _, llm = _compose([])
    answer = service.ask("q")
    assert answer.evidence_count == 0
    assert "No evidence was retrieved." in llm.calls[0][0][1].content
