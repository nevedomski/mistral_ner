"""Tests for api/serve.py module."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.serve import Entity, HealthResponse, NERRequest, NERResponse, app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_loaded():
    """Mock model being loaded."""
    with (
        patch("api.serve.model", Mock()),
        patch("api.serve.tokenizer", Mock()),
        patch("api.serve.config", Mock()) as mock_config,
    ):
        mock_config.data.label_names = ["O", "B-PER", "I-PER"]
        yield


@pytest.fixture
def mock_model_not_loaded():
    """Mock model not being loaded."""
    with patch("api.serve.model", None), patch("api.serve.tokenizer", None), patch("api.serve.config", None):
        yield


class TestPydanticModels:
    """Test Pydantic model definitions."""

    def test_ner_request_model(self):
        """Test NERRequest model validation."""
        # Valid request
        request = NERRequest(texts=["Hello world"], batch_size=2)
        assert request.texts == ["Hello world"]
        assert request.batch_size == 2

        # Default batch_size
        request = NERRequest(texts=["Hello world"])
        assert request.batch_size == 1

    def test_entity_model(self):
        """Test Entity model validation."""
        entity = Entity(text="John", type="PER", start=0, end=1)
        assert entity.text == "John"
        assert entity.type == "PER"
        assert entity.start == 0
        assert entity.end == 1

    def test_ner_response_model(self):
        """Test NERResponse model validation."""
        entities = [Entity(text="John", type="PER", start=0, end=1)]
        response = NERResponse(text="John works", words=["John", "works"], labels=["B-PER", "O"], entities=entities)
        assert response.text == "John works"
        assert len(response.entities) == 1

    def test_health_response_model(self):
        """Test HealthResponse model validation."""
        response = HealthResponse(status="healthy", model_loaded=True, device="cuda")
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.device == "cuda"


class TestStartupEvent:
    """Test application startup event."""

    @patch("api.serve.Config.from_yaml")
    @patch("api.serve.setup_logging")
    @patch("api.serve.load_model_for_inference")
    @patch.dict(os.environ, {"NER_MODEL_PATH": "/test/model", "NER_BASE_MODEL": "test-base"})
    async def test_startup_event_success(self, mock_load_model, mock_setup_logging, mock_config_from_yaml):
        """Test successful startup event."""
        # Import the startup function
        from api.serve import startup_event

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_config = Mock()
        mock_config_from_yaml.return_value = mock_config

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        with (
            patch("api.serve.model", None),
            patch("api.serve.tokenizer", None),
            patch("api.serve.config", None),
        ):
            await startup_event()

        mock_load_model.assert_called_once_with(model_path="/test/model", base_model="test-base", config=mock_config)

    @patch("api.serve.Config.from_yaml")
    @patch("api.serve.setup_logging")
    @patch("api.serve.load_model_for_inference")
    async def test_startup_event_default_paths(self, mock_load_model, mock_setup_logging, mock_config_from_yaml):
        """Test startup event with default model paths."""
        from api.serve import startup_event

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_config = Mock()
        mock_config_from_yaml.return_value = mock_config

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Clear environment variables
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("api.serve.model", None),
            patch("api.serve.tokenizer", None),
            patch("api.serve.config", None),
        ):
            await startup_event()

        mock_load_model.assert_called_once_with(
            model_path="./mistral-ner-finetuned-final", base_model="mistralai/Mistral-7B-v0.3", config=mock_config
        )

    @patch("api.serve.Config.from_yaml")
    @patch("api.serve.setup_logging")
    @patch("api.serve.load_model_for_inference")
    async def test_startup_event_model_load_failure(self, mock_load_model, mock_setup_logging, mock_config_from_yaml):
        """Test startup event with model loading failure."""
        from api.serve import startup_event

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_config = Mock()
        mock_config_from_yaml.return_value = mock_config

        mock_load_model.side_effect = Exception("Model load failed")

        with patch("api.serve.model", None), patch("api.serve.tokenizer", None), patch("api.serve.config", None):
            # Should not raise exception, just log error
            await startup_event()

        mock_logger.error.assert_called()


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_model_loaded(self, client, mock_model_loaded):
        """Test health check when model is loaded."""
        with patch("api.serve.device", "cuda"):
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["device"] == "cuda"

    def test_health_check_model_not_loaded(self, client, mock_model_not_loaded):
        """Test health check when model is not loaded."""
        with patch("api.serve.device", "cpu"):
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert data["device"] == "cpu"


class TestPredictEndpoint:
    """Test prediction endpoint."""

    @patch("api.serve.predict_entities")
    def test_predict_success(self, mock_predict_entities, client, mock_model_loaded):
        """Test successful prediction."""
        mock_predictions = [
            {
                "text": "John works at Microsoft",
                "words": ["John", "works", "at", "Microsoft"],
                "labels": ["B-PER", "O", "O", "B-ORG"],
                "entities": [
                    {"text": "John", "type": "PER", "start": 0, "end": 1},
                    {"text": "Microsoft", "type": "ORG", "start": 3, "end": 4},
                ],
            }
        ]
        mock_predict_entities.return_value = mock_predictions

        request_data = {"texts": ["John works at Microsoft"], "batch_size": 1}

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["text"] == "John works at Microsoft"
        assert len(data[0]["entities"]) == 2

    def test_predict_model_not_loaded(self, client, mock_model_not_loaded):
        """Test prediction when model is not loaded."""
        request_data = {"texts": ["Test text"], "batch_size": 1}

        response = client.post("/predict", json=request_data)

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_no_texts(self, client, mock_model_loaded):
        """Test prediction with no texts provided."""
        request_data = {"texts": [], "batch_size": 1}

        response = client.post("/predict", json=request_data)

        assert response.status_code == 400
        assert "No texts provided" in response.json()["detail"]

    def test_predict_too_many_texts(self, client, mock_model_loaded):
        """Test prediction with too many texts."""
        request_data = {
            "texts": ["text"] * 101,  # More than 100 texts
            "batch_size": 1,
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 400
        assert "Maximum 100 texts per request" in response.json()["detail"]

    @patch("api.serve.predict_entities")
    def test_predict_with_batch_size(self, mock_predict_entities, client, mock_model_loaded):
        """Test prediction with custom batch size."""
        mock_predictions = [
            {"text": "Text 1", "words": ["Text", "1"], "labels": ["O", "O"], "entities": []},
            {"text": "Text 2", "words": ["Text", "2"], "labels": ["O", "O"], "entities": []},
        ]
        mock_predict_entities.return_value = mock_predictions

        request_data = {"texts": ["Text 1", "Text 2"], "batch_size": 2}

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        mock_predict_entities.assert_called_once()
        # Verify batch_size was passed correctly
        call_kwargs = mock_predict_entities.call_args[1]
        assert call_kwargs["batch_size"] == 2

    @patch("api.serve.predict_entities")
    def test_predict_default_batch_size(self, mock_predict_entities, client, mock_model_loaded):
        """Test prediction with default batch size."""
        mock_predictions = [{"text": "Test text", "words": ["Test", "text"], "labels": ["O", "O"], "entities": []}]
        mock_predict_entities.return_value = mock_predictions

        request_data = {
            "texts": ["Test text"]
            # No batch_size specified
        }

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        call_kwargs = mock_predict_entities.call_args[1]
        assert call_kwargs["batch_size"] == 1

    @patch("api.serve.predict_entities")
    def test_predict_prediction_error(self, mock_predict_entities, client, mock_model_loaded):
        """Test prediction with prediction error."""
        mock_predict_entities.side_effect = Exception("Prediction failed")

        request_data = {"texts": ["Test text"], "batch_size": 1}

        with patch("api.serve.logger", Mock()) as mock_logger:
            response = client.post("/predict", json=request_data)

        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]
        mock_logger.error.assert_called()


class TestPredictSimpleEndpoint:
    """Test simple prediction endpoint."""

    @patch("api.serve.predict_entities")
    def test_predict_simple_success(self, mock_predict_entities, client, mock_model_loaded):
        """Test successful simple prediction."""
        mock_predictions = [
            {
                "text": "John works at Microsoft",
                "words": ["John", "works", "at", "Microsoft"],
                "labels": ["B-PER", "O", "O", "B-ORG"],
                "entities": [
                    {"text": "John", "type": "PER", "start": 0, "end": 1},
                    {"text": "Microsoft", "type": "ORG", "start": 3, "end": 4},
                ],
            }
        ]
        mock_predict_entities.return_value = mock_predictions

        texts = ["John works at Microsoft"]
        response = client.post("/predict_simple", json=texts)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["text"] == "John works at Microsoft"
        assert "entities" in data[0]
        # Should not include words and labels (simplified format)
        assert "words" not in data[0]
        assert "labels" not in data[0]

    def test_predict_simple_model_not_loaded(self, client, mock_model_not_loaded):
        """Test simple prediction when model is not loaded."""
        texts = ["Test text"]
        response = client.post("/predict_simple", json=texts)

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    @patch("api.serve.predict_entities")
    def test_predict_simple_prediction_error(self, mock_predict_entities, client, mock_model_loaded):
        """Test simple prediction with prediction error."""
        mock_predict_entities.side_effect = Exception("Prediction failed")

        texts = ["Test text"]

        with patch("api.serve.logger", Mock()) as mock_logger:
            response = client.post("/predict_simple", json=texts)

        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]
        mock_logger.error.assert_called()

    @patch("api.serve.predict_entities")
    def test_predict_simple_multiple_texts(self, mock_predict_entities, client, mock_model_loaded):
        """Test simple prediction with multiple texts."""
        mock_predictions = [{"text": "Text 1", "entities": []}, {"text": "Text 2", "entities": []}]
        mock_predict_entities.return_value = mock_predictions

        texts = ["Text 1", "Text 2"]
        response = client.post("/predict_simple", json=texts)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

        # Verify batch_size is always 1 for simple endpoint
        call_kwargs = mock_predict_entities.call_args[1]
        assert call_kwargs["batch_size"] == 1


class TestMainFunction:
    """Test main function for running the server."""

    @patch("api.serve.uvicorn.run")
    def test_main_default_args(self, mock_uvicorn_run):
        """Test main function with default arguments."""
        from api.serve import main

        with patch("sys.argv", ["serve.py"]):
            main()

        mock_uvicorn_run.assert_called_once_with("api.serve:app", host="0.0.0.0", port=8000, reload=False)

    @patch("api.serve.uvicorn.run")
    def test_main_custom_args(self, mock_uvicorn_run):
        """Test main function with custom arguments."""
        from api.serve import main

        test_argv = ["serve.py", "--host", "127.0.0.1", "--port", "9000", "--reload"]
        with patch("sys.argv", test_argv):
            main()

        mock_uvicorn_run.assert_called_once_with("api.serve:app", host="127.0.0.1", port=9000, reload=True)


class TestGlobalVariables:
    """Test global variable initialization."""

    def test_global_variables_default_values(self):
        """Test that global variables have expected default values."""
        # Import to ensure module is loaded
        import api.serve

        # Check that app is FastAPI instance
        assert hasattr(api.serve.app, "title")
        assert api.serve.app.title == "Mistral NER API"

        # Check device selection logic

        assert api.serve.device in ["cuda", "cpu"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
