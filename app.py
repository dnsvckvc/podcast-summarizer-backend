import os
import uuid
import threading

from typing import Dict, Any
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from task_manager import TaskStatus, TaskManager
from models.downloaders.yt_downloader import YTDownloader
from utils.validators import URLValidator, InputValidator
from models.summarizers.openai_summarizer import OpenAI_Summarizer
from models.transcribers.salad_transcriber import SaladTranscriber
from utils.app_utils import load_config, setup_logger, copy_cookies
from models.transcribers.whisper_transcriber import WhisperTranscriber
from models.downloaders.rss_feed_downloader import RSS_Feed_Downloader

# Load environment variables and configuration
load_dotenv(override=True)
config = load_config()
logger = setup_logger()

# Copy YouTube cookies to temporary location
copy_cookies(config)

# Initialize components
task_manager = TaskManager()

# Initialize downloaders
yt_downloader = YTDownloader(config=config["youtube"])
rss_downloader = RSS_Feed_Downloader(config=config["rss_feed"])

# Initialize transcribers
transcriber_type = config.get("transcriber", "salad")
if transcriber_type == "salad":
    transcriber = SaladTranscriber(config=config["salad"])
else:
    transcriber = WhisperTranscriber(config=config["whisper"])

# Initialize summarizer
summarizer = OpenAI_Summarizer(config=config["openai"])

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")])


def validate_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate incoming request data.

    Args:
        data (Dict[str, Any]): Request data to validate

    Returns:
        Dict[str, Any]: Validation result with 'valid' boolean and optional 'errors'
    """
    errors = []

    # Validate required fields
    source_url = data.get("source_url")
    platform = data.get("platform")

    if not source_url:
        errors.append("source_url is required")
    if not platform:
        errors.append("platform is required")

    if errors:
        return {"valid": False, "errors": errors}

    # Validate platform
    platform_validation = InputValidator.validate_platform(platform)
    if not platform_validation["valid"]:
        errors.append(platform_validation["error"])
        return {"valid": False, "errors": errors}

    platform = platform_validation["value"]

    # Validate URL based on platform
    if platform == "youtube":
        url_validation = URLValidator.validate_youtube_url(source_url)
    else:  # rss
        url_validation = URLValidator.validate_rss_url(source_url)

    if not url_validation["valid"]:
        errors.append(
            f"Invalid {platform} URL: {url_validation.get('error', 'Unknown error')}"
        )

    # Validate episode name
    episode_name = data.get("episode_name")
    episode_validation = InputValidator.validate_episode_name(episode_name, platform)
    if not episode_validation["valid"]:
        errors.append(episode_validation["error"])

    # Validate detail level
    detail_level = data.get("detail_level", 0.5)
    detail_validation = InputValidator.validate_detail_level(detail_level)
    if not detail_validation["valid"]:
        errors.append(detail_validation["error"])

    if errors:
        return {"valid": False, "errors": errors}

    return {
        "valid": True,
        "data": {
            "source_url": source_url,
            "platform": platform,
            "episode_name": episode_validation["value"],
            "detail_level": detail_validation["value"],
        },
    }


def process_podcast(
    task_id: str, source_url: str, episode_name: str, detail_level: float, platform: str
) -> None:
    """
    Process podcast summarization asynchronously.

    Args:
        task_id (str): Unique task identifier
        source_url (str): URL of the podcast source
        episode_name (str): Name of the episode (for RSS feeds)
        detail_level (float): Summary detail level (0.0-1.0)
        platform (str): Platform type ('youtube' or 'rss')
    """
    try:
        # Select appropriate downloader
        downloader = yt_downloader if platform == "youtube" else rss_downloader

        # Step 1: Download/Process audio
        task_manager.update_task(
            task_id,
            status=TaskStatus.DOWNLOADING,
            progress=10.0,
            message="Processing audio source...",
        )
        import json

        audio_path, metadata = downloader.download_episode(source_url, episode_name)
        logger.info(f"Processed audio for: {metadata.get('title', 'Unknown')}")

        # Step 2: Transcribe audio
        task_manager.update_task(
            task_id,
            status=TaskStatus.TRANSCRIBING,
            progress=30.0,
            message="Transcribing audio to text...",
        )

        transcription = transcriber.transcribe(
            audio_path=audio_path, video_id=metadata["video_id"]
        )
        logger.info("Transcription completed successfully")

        # Step 3: Generate summary
        task_manager.update_task(
            task_id,
            status=TaskStatus.SUMMARIZING,
            progress=70.0,
            message="Generating AI summary...",
        )

        summary = summarizer.summarize(transcription, detail=detail_level)
        logger.info("Summary generation completed")

        # Complete the task
        result = {
            "title": metadata.get("title", "Unknown Title"),
            "summary": summary,
            "thumbnail": metadata.get("thumbnail"),
            "channel": metadata.get("channel", "Unknown Channel"),
            "duration_string": metadata.get("duration_string", "Unknown"),
            "release_date": metadata.get("release_date", "Unknown"),
        }

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0,
            message="Summary completed successfully!",
            result=result,
        )

        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        error_message = str(e)
        logger.exception(f"Error processing task {task_id}: {error_message}")

        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            progress=0.0,
            message="Processing failed",
            error=error_message,
        )


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    """Health check endpoint."""
    return (
        jsonify(
            {
                "message": "Podcast Summarizer API",
                "version": "2.0.0",
                "status": "healthy",
            }
        ),
        200,
    )


@app.route("/api/validate", methods=["POST"])
@cross_origin()
def validate_url():
    """
    Validate URL endpoint for frontend validation.

    Expected JSON:
        {
            "url": "string",
            "platform": "youtube" | "rss"
        }

    Returns:
        JSON with validation result and optional metadata
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Request body is required"}), 400

        url = data.get("url")
        platform = data.get("platform")

        if not url or not platform:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Both 'url' and 'platform' are required",
                    }
                ),
                400,
            )

        if platform == "youtube":
            validation_result = URLValidator.validate_youtube_url(url)
        elif platform == "rss":
            validation_result = URLValidator.validate_rss_url(url)
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Platform must be either 'youtube' or 'rss'",
                    }
                ),
                400,
            )

        if validation_result["valid"]:
            return (
                jsonify(
                    {
                        "success": True,
                        "message": f"Valid {platform} URL",
                        "data": validation_result,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": validation_result.get("error", "Invalid URL"),
                    }
                ),
                400,
            )

    except Exception as e:
        logger.exception("Error in URL validation")
        return (
            jsonify(
                {"success": False, "error": "Internal server error during validation"}
            ),
            500,
        )


@app.route("/api/summarize", methods=["POST"])
@cross_origin()
def summarize_endpoint():
    """
    Start asynchronous podcast summarization.

    Expected JSON:
        {
            "source_url": "string",
            "episode_name": "string" | null,
            "detail_level": float (0.0-1.0),
            "platform": "youtube" | "rss"
        }

    Returns:
        JSON with task_id for status tracking
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Request body is required"}), 400

        validation_result = validate_request_data(data)
        if not validation_result["valid"]:
            return (
                jsonify({"success": False, "errors": validation_result["errors"]}),
                400,
            )

        validated_data = validation_result["data"]

        task_id = str(uuid.uuid4())
        task_manager.create_task(task_id)

        threading.Thread(
            target=process_podcast,
            args=(
                task_id,
                validated_data["source_url"],
                validated_data["episode_name"],
                validated_data["detail_level"],
                validated_data["platform"],
            ),
            daemon=True,
        ).start()

        logger.info(
            f"Started processing task {task_id} for {validated_data['platform']} content"
        )

        return (
            jsonify(
                {
                    "success": True,
                    "task_id": task_id,
                    "message": "Processing started successfully",
                }
            ),
            202,
        )

    except Exception as e:
        logger.exception("Error starting summarization task")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/status/<task_id>", methods=["GET"])
@cross_origin()
def get_task_status(task_id: str):
    """
    Get current status of a summarization task.

    Args:
        task_id (str): Task identifier

    Returns:
        JSON with task status and progress information
    """
    try:
        task_info = task_manager.get_task_dict(task_id)

        if not task_info:
            return jsonify({"success": False, "error": "Task not found"}), 404

        return jsonify({"success": True, "task": task_info}), 200

    except Exception as e:
        logger.exception(f"Error getting task status for {task_id}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/result/<task_id>", methods=["GET"])
@cross_origin()
def get_task_result(task_id: str):
    """
    Get result of a completed summarization task.

    Args:
        task_id (str): Task identifier

    Returns:
        JSON with task result or error information
    """
    try:
        task_info = task_manager.get_task(task_id)

        if not task_info:
            return jsonify({"success": False, "error": "Task not found"}), 404

        if task_info.status == TaskStatus.COMPLETED:
            return jsonify({"success": True, "result": task_info.result}), 200
        elif task_info.status == TaskStatus.FAILED:
            return (
                jsonify({"success": False, "error": task_info.error or "Task failed"}),
                500,
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Task not completed yet",
                        "status": task_info.status.value,
                    }
                ),
                202,
            )

    except Exception as e:
        logger.exception(f"Error getting task result for {task_id}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.exception("Internal server error")
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    task_manager.cleanup_old_tasks()
    app.run()
