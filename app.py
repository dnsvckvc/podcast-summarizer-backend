from curses import meta
import os
import uuid

from typing import Dict, Any
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from concurrent.futures import ThreadPoolExecutor
from models.managers.auth_manager import AuthManager
from utils.validators import URLValidator, InputValidator
from flask_jwt_extended import jwt_required, get_jwt, JWTManager
from models.managers.task_manager import TaskStatus, TaskManager
from utils.app_utils import load_config, setup_logger, copy_cookies


# Load environment variables and configuration
load_dotenv(override=True)
config = load_config()
logger = setup_logger()
copy_cookies(config)

# Initialize components
auth_manager = AuthManager()
task_manager = TaskManager(config)
executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 5))


# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")])
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "secretKey")
jwt_manager = JWTManager(app)
VERBOSE = True


def _get_jti():
    return get_jwt()["jti"]


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
    user_id: str,
    task_id: str,
    source_url: str,
    episode_name: str,
    detail_level: float,
    platform: str,
) -> None:
    """Downloads, transcribes and summarizes using user-specific context"""
    try:
        ctx = task_manager.get_user_context(user_id)
        downloader = ctx.yt_downloader if platform == "youtube" else ctx.rss_downloader
        transcriber = ctx.transcriber
        summarizer = ctx.summarizer

        # Step 1: Download/Process audio
        task_manager.update_task(
            user_id,
            task_id,
            status=TaskStatus.DOWNLOADING,
            progress=10.0,
            message="Downloading audio...",
        )

        audio_path, metadata = downloader.download_episode(source_url, episode_name)

        if VERBOSE:
            logger.info(f"Processed audio for: {metadata.get('title', 'Unknown')}")

        # Step 2: Transcribe audio
        task_manager.update_task(
            user_id,
            task_id,
            status=TaskStatus.TRANSCRIBING,
            progress=30.0,
            message="Transcribing audio...",
        )

        transcription = transcriber.transcribe(
            audio_path=audio_path, video_id=metadata["video_id"]
        )

        if VERBOSE:
            logger.info("Transcription completed successfully")

        # Step 3: Generate summary
        task_manager.update_task(
            user_id,
            task_id,
            status=TaskStatus.SUMMARIZING,
            progress=70.0,
            message="Generating summary...",
        )

        summary = summarizer.summarize(transcription, detail=detail_level)

        if VERBOSE:
            logger.info("Summary generation completed")

        # Complete the task
        result = {
            "title": metadata.get("title", "Unknown Title"),
            "summary": summary,
            "thumbnail": metadata.get("thumbnail"),
            "channel": metadata.get("channel", "Unknown Channel"),
            "duration_string": metadata.get("duration_string", "Unknown"),
            "release_date": metadata.get("release_date", "Unknown"),
            "transcript": transcription,
        }

        task_manager.update_task(
            user_id,
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0,
            message="Done.",
            result=result,
        )

        if VERBOSE:
            logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.exception("Task %s for user %s failed", task_id, user_id)
        task_manager.update_task(
            user_id,
            task_id,
            status=TaskStatus.FAILED,
            progress=0.0,
            message="Failed",
            error=str(e),
        )


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    """Health check endpoint."""
    return (
        jsonify(
            {
                "message": "Podcast Summarizer API",
                "version": "1.2.0",
                "status": "healthy",
            }
        ),
        200,
    )


@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
@cross_origin()
def login():
    """Authenticate user and return JWT token"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Request body required"}), 400

        username = data.get("username", "").strip()
        password = data.get("password", "")

        if not username or not password:
            return (
                jsonify({"success": False, "error": "Username and password required"}),
                400,
            )

        user_data = auth_manager.authenticate_user(username, password)
        if not user_data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid username or password. Please try again.",
                    }
                ),
                401,
            )

        token = auth_manager.create_token(username)

        return jsonify(
            {
                "success": True,
                "token": token,
                "user": {"username": username, "role": user_data["role"]},
                "message": "Login successful",
            }
        )

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/validate", methods=["POST"])
@cross_origin()
@jwt_required()
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
        logger.exception(f"Error in URL validation - {e}")
        return (
            jsonify(
                {"success": False, "error": "Internal server error during validation"}
            ),
            500,
        )


@app.route("/api/summarize", methods=["POST"])
@cross_origin()
@jwt_required()
def summarize_endpoint():
    """
    Start podcast summarization.

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

        user_id = _get_jti()  # Use JWT ID as user identifier
        task_id = str(uuid.uuid4())

        task_manager.create_task(user_id, task_id)

        executor.submit(
            process_podcast,
            user_id,
            task_id,
            validated_data["source_url"],
            validated_data["episode_name"],
            validated_data["detail_level"],
            validated_data["platform"],
        )

        if VERBOSE:
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
        logger.exception(f"Error starting summarization task - {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/status/<task_id>", methods=["GET"])
@cross_origin()
@jwt_required()
def get_task_status(task_id: str):
    """
    Get current status of a summarization task.

    Args:
        task_id (str): Task identifier

    Returns:
        JSON with task status and progress information
    """
    try:
        task_info = task_manager.get_task_dict(_get_jti(), task_id)

        if not task_info:
            return jsonify({"success": False, "error": "Task not found"}), 404

        return jsonify({"success": True, "task": task_info}), 200

    except Exception as e:
        logger.exception(f"Error getting task status for {task_id} - {e}")
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
