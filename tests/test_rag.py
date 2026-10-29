import unittest
from unittest.mock import patch

from rag_assistant.documents import DocumentChunk, extract_chunks
from rag_assistant.generation import generate_answer
from rag_assistant.retrieval import GeminiEmbedder, VectorIndex


class FakeEmbedder:
    def embed_documents(self, texts):
        return [
            [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, query):
        return [1.0, 0.0] if "python" in query.lower() else [0.0, 1.0]


class DocumentChunkingTests(unittest.TestCase):
    def test_text_is_split_with_overlap_and_source_metadata(self):
        text = ("alpha beta gamma delta " * 8).encode()
        chunks = extract_chunks("guide.txt", text, chunk_size=40, chunk_overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.source == "guide.txt" for chunk in chunks))
        self.assertTrue(all(chunk.page is None for chunk in chunks))
        self.assertTrue(all(len(chunk.content) <= 40 for chunk in chunks))
        self.assertIn(chunks[0].content[-10:], chunks[1].content)

    def test_rejects_unsupported_document_types(self):
        with self.assertRaises(ValueError):
            extract_chunks("guide.docx", b"text")

    def test_rejects_invalid_chunk_configuration(self):
        with self.assertRaises(ValueError):
            extract_chunks("guide.txt", b"text", chunk_size=10, chunk_overlap=10)


class VectorSearchTests(unittest.TestCase):
    def test_search_ranks_matching_passages_and_limits_results(self):
        index = VectorIndex(FakeEmbedder())
        index.add_documents(
            [
                DocumentChunk("Python deployment guide", "python.md"),
                DocumentChunk("Database migration notes", "database.md"),
            ]
        )

        results = index.search("python deployment", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.source, "python.md")
        self.assertEqual(results[0].score, 1.0)

    def test_search_validates_query_and_result_count(self):
        index = VectorIndex(FakeEmbedder())
        with self.assertRaises(ValueError):
            index.search(" ")
        with self.assertRaises(ValueError):
            index.search("query", top_k=0)


class EmbeddingTests(unittest.TestCase):
    @patch("rag_assistant.retrieval.genai.Client")
    def test_gemini_embeddings_are_requested_per_passage(self, client):
        client.return_value.models.embed_content.side_effect = [
            type(
                "Response",
                (),
                {
                    "embeddings": [type("Item", (), {"values": vector})()]
                },
            )()
            for vector in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5])
        ]
        embedder = GeminiEmbedder("test-key")

        vectors = embedder.embed_documents(["first", "second", "third"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        self.assertEqual(client.return_value.models.embed_content.call_count, 3)
        self.assertEqual(
            client.return_value.models.embed_content.call_args_list[0].kwargs["model"],
            "gemini-embedding-2",
        )
        self.assertIn(
            "title: none | text: first",
            client.return_value.models.embed_content.call_args_list[0].kwargs["contents"],
        )

    @patch("rag_assistant.retrieval.genai.Client")
    def test_query_embedding_uses_question_answering_task_prefix(self, client):
        client.return_value.models.embed_content.return_value.embeddings = [
            type("Item", (), {"values": [0.1, 0.9]})()
        ]
        embedder = GeminiEmbedder("test-key")

        vector = embedder.embed_query("How does this work?")

        self.assertEqual(vector, [0.1, 0.9])
        request = client.return_value.models.embed_content.call_args.kwargs
        self.assertEqual(
            request["contents"],
            "task: question answering | query: How does this work?",
        )

    @patch("rag_assistant.retrieval.genai.Client")
    def test_gemini_embedder_requires_api_key(self, client):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            GeminiEmbedder("")
        client.assert_not_called()


class AnswerGenerationTests(unittest.TestCase):
    def test_requires_api_key_when_passages_are_available(self):
        passage = DocumentChunk("Python uses indentation.", "guide.md")

        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            generate_answer(
                "How does Python structure code?",
                [type("Result", (), {"chunk": passage, "score": 1.0})()],
                "",
            )

    def test_returns_grounded_fallback_without_passages(self):
        answer = generate_answer("What is in the docs?", [], "")

        self.assertIn("couldn't find relevant information", answer)

    @patch("rag_assistant.generation.genai.Client")
    def test_sends_labeled_evidence_and_returns_gemini_answer(self, client):
        client.return_value.models.generate_content.return_value.text = (
            "Python uses indentation. [1]"
        )
        passage = DocumentChunk("Python uses indentation.", "guide.md", page=3)
        result = type("Result", (), {"chunk": passage, "score": 1.0})()

        answer = generate_answer("How does Python structure code?", [result], "test-key")

        self.assertEqual(answer, "Python uses indentation. [1]")
        request = client.return_value.models.generate_content.call_args.kwargs
        self.assertIn(
            "[1] Source: guide.md, page 3\nPython uses indentation.",
            request["contents"],
        )
        self.assertEqual(request["model"], "gemini-3.5-flash-lite")
        self.assertEqual(request["config"].temperature, 0)


if __name__ == "__main__":
    unittest.main()
