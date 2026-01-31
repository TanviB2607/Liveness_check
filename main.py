# from flask import Flask, jsonify, send_from_directory, request
# from flask_socketio import SocketIO, emit
# import os
# import time
# import numpy as np
# import cv2

# from services.liveness_service import LivenessService, ActiveLivenessTracker

# app = Flask(__name__)
# app.config["SECRET_KEY"] = "dev"

# # ✅ threading mode = simplest on Windows (no eventlet)
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# PORT = int(os.getenv("FLASK_PORT", "5002"))

# live_service = LivenessService()

# # per-client sessions
# SESSIONS = {}
# TTL_SECONDS = 60

# def instruction_for(state: str) -> str:
#     if state == "WAIT_LEFT":  return "Look LEFT 👈 and hold for a second"
#     if state == "WAIT_RIGHT": return "Now look RIGHT 👉 and hold for a second"
#     if state == "WAIT_BLINK": return "Now BLINK once 👁️"
#     return "Follow the instructions"

# def decode_jpeg_to_bgr(jpeg_bytes: bytes):
#     arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     return img

# @app.get("/")
# def index():
#     return send_from_directory(".", "upload.html")

# @app.get("/health")
# def health():
#     return jsonify({"ok": True, "port": PORT})

# @socketio.on("connect")
# def ws_connect():
#     emit("server_update", {
#         "status": "IN_PROGRESS",
#         "state": "WAIT_LEFT",
#         "instruction": "Connected ✅ Click Start & Record",
#     })

# @socketio.on("liveness_start")
# def ws_start():
#     sid = request.sid
#     SESSIONS[sid] = {
#         "tracker": ActiveLivenessTracker(live_service.face_mesh, debug=False),
#         "frames_seen": 0,
#         "last_seen": time.time(),
#         "passed": False,
#     }
#     tr = SESSIONS[sid]["tracker"]
#     emit("server_update", {
#         "status": "IN_PROGRESS",
#         "state": tr.state,
#         "instruction": instruction_for(tr.state),
#         "progress": {"frames_seen": 0, "blink_count": 0}
#     })

# @socketio.on("liveness_frame")
# def ws_frame(jpeg_bytes):
#     sid = request.sid
#     sess = SESSIONS.get(sid)
#     if not sess:
#         emit("server_update", {"status": "FAILED", "error": "session_missing"})
#         return

#     now = time.time()
#     if now - sess["last_seen"] > TTL_SECONDS:
#         SESSIONS.pop(sid, None)
#         emit("server_update", {"status": "FAILED", "error": "session_expired"})
#         return
#     sess["last_seen"] = now

#     if isinstance(jpeg_bytes, bytearray):
#         jpeg_bytes = bytes(jpeg_bytes)

#     frame = decode_jpeg_to_bgr(jpeg_bytes)
#     if frame is None:
#         emit("server_update", {"status": "IN_PROGRESS", "error": "bad_frame"})
#         return

#     i = sess["frames_seen"]
#     sess["frames_seen"] += 1

#     tracker: ActiveLivenessTracker = sess["tracker"]
#     status, meta = tracker.update(frame, i)

#     if status == "PASSED":
#         sess["passed"] = True
#         emit("server_update", {
#             "status": "PASSED",
#             "state": tracker.state,
#             "instruction": "✅ Liveness PASSED",
#             "progress": {
#                 "frames_seen": sess["frames_seen"],
#                 "blink_count": meta.get("blink_count", 0),
#                 "left_at": meta.get("left_at"),
#                 "right_at": meta.get("right_at"),
#             }
#         })
#         return

#     emit("server_update", {
#         "status": "IN_PROGRESS",
#         "state": tracker.state,
#         "instruction": instruction_for(tracker.state),
#         "progress": {
#             "frames_seen": sess["frames_seen"],
#             "blink_count": meta.get("blink_count", 0),
#             "left_at": meta.get("left_at"),
#             "right_at": meta.get("right_at"),
#         }
#     })

# @socketio.on("liveness_finish")
# def ws_finish():
#     sid = request.sid
#     sess = SESSIONS.get(sid)
#     if not sess:
#         emit("server_update", {"status": "FAILED", "error": "session_missing"})
#         return

#     if sess.get("passed"):
#         emit("server_update", {"status": "PASSED", "instruction": "✅ Already passed"})
#     else:
#         status, meta = sess["tracker"].finalize()
#         emit("server_update", {"status": "FAILED", "instruction": "❌ Liveness FAILED", "meta": meta})

#     SESSIONS.pop(sid, None)

# @socketio.on("disconnect")
# def ws_disconnect():
#     SESSIONS.pop(request.sid, None)

# if __name__ == "__main__":
#     socketio.run(app, host="127.0.0.1", port=PORT, debug=False)



# OG CODE
# from flask import Flask, jsonify, send_from_directory, request
# from flask_socketio import SocketIO, emit
# import os
# import time
# import numpy as np
# import cv2

# from services.liveness_service import LivenessService, ActiveLivenessTracker

# # --------------------------------------------------
# # App setup
# # --------------------------------------------------

# app = Flask(__name__)
# app.config["SECRET_KEY"] = "dev"

# socketio = SocketIO(
#     app,
#     cors_allowed_origins="*",
#     async_mode="threading"  # Windows safe
# )

# PORT = int(os.getenv("FLASK_PORT", "5002"))

# live_service = LivenessService()

# # per-client sessions
# SESSIONS = {}
# TTL_SECONDS = 60

# # --------------------------------------------------
# # Helpers
# # --------------------------------------------------

# def instruction_for(action: str):
#     if action == "LEFT":
#         return "Turn your head LEFT 👈"
#     if action == "RIGHT":
#         return "Turn your head RIGHT 👉"
#     if action == "BLINK":
#         return "Blink your eyes 👁️"
#     if action == "SMILE":
#         return "Smile 🙂"
#     if action == "NOD":
#         return "Nod your head up & down ⬆️⬇️"
#     return "Follow the instructions"


# def decode_jpeg_to_bgr(jpeg_bytes: bytes):
#     arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     return img


# def is_face_aligned(frame):
#     h, w, _ = frame.shape
#     cx, cy = w // 2, h // 2

#     margin_x = w * 0.15
#     margin_y = h * 0.15

#     return (
#         cx > margin_x and cx < w - margin_x
#         and cy > margin_y and cy < h - margin_y
#     )

# # --------------------------------------------------
# # Routes
# # --------------------------------------------------

# @app.get("/")
# def index():
#     return send_from_directory(".", "upload.html")


# @app.get("/health")
# def health():
#     return jsonify({"ok": True, "port": PORT})

# # --------------------------------------------------
# # WebSocket
# # --------------------------------------------------

# @socketio.on("connect")
# def ws_connect():
#     emit(
#         "server_update",
#         {
#             "status": "IN_PROGRESS",
#             "action": "START",
#             "instruction": "Connected ✅ Click Start",
#         },
#     )


# @socketio.on("liveness_start")
# def ws_start():
#     sid = request.sid

#     tracker = ActiveLivenessTracker(live_service.face_mesh)
#     SESSIONS[sid] = {
#         "tracker": tracker,
#         "frames_seen": 0,
#         "last_seen": time.time(),
#         "passed": False,
#     }

#     emit(
#         "server_update",
#         {
#             "status": "IN_PROGRESS",
#             "action": tracker.current_action,
#             "instruction": instruction_for(tracker.current_action),
#         },
#     )


# @socketio.on("liveness_frame")
# def ws_frame(jpeg_bytes):
#     sid = request.sid
#     sess = SESSIONS.get(sid)

#     if not sess:
#         return

#     now = time.time()

#     # ---------------- GREEN BUFFER HOLD ----------------
#     if sess.get("hold_green"):
#         if now - sess["last_action_time"] < 3:
#             emit(
#                 "server_update",
#                 {
#                     "status": "IN_PROGRESS",
#                     "instruction": "✅ Action completed!",
#                     "face_aligned": True,
#                 },
#             )
#             return
#         else:
#             sess["hold_green"] = False

#             tracker = sess["tracker"]

#             # ✅ ALL ACTIONS DONE
#             if tracker.current_action is None:
#                 emit(
#                     "server_update",
#                     {
#                         "status": "PASSED",
#                         "instruction": "✅ Liveness PASSED Successfully",
#                         "face_aligned": True,
#                     },
#                 )
#                 SESSIONS.pop(sid, None)
#                 return

#             # ➡️ NEXT ACTION EXISTS
#             emit(
#                 "server_update",
#                 {
#                     "status": "IN_PROGRESS",
#                     "instruction": instruction_for(tracker.current_action),
#                     "face_aligned": False,
#                 },
#             )
#             return

#     # ---------------- FRAME DECODE ----------------
#     if isinstance(jpeg_bytes, bytearray):
#         jpeg_bytes = bytes(jpeg_bytes)

#     frame = decode_jpeg_to_bgr(jpeg_bytes)
#     if frame is None:
#         return

#     i = sess["frames_seen"]
#     sess["frames_seen"] += 1

#     tracker = sess["tracker"]
#     status, meta = tracker.update(frame, i)

#     # ---------------- FAILED ----------------
#     if status == "FAILED":
#         emit(
#             "server_update",
#             {
#                 "status": "FAILED",
#                 "instruction": "❌ Time up! Action not performed",
#                 "reason": meta,
#             },
#         )
#         SESSIONS.pop(sid, None)
#         return

#     # ---------------- PASSED ----------------
#     if status == "PASSED":
#         emit(
#             "server_update",
#             {
#                 "status": "PASSED",
#                 "instruction": "✅ Liveness PASSED Successfully",
#                 "face_aligned": True,
#             },
#         )
#         SESSIONS.pop(sid, None)
#         return

#     # ---------------- ACTION COMPLETED ----------------
#     if meta.get("action_done"):
#         sess["hold_green"] = True
#         sess["last_action_time"] = now

#         emit(
#             "server_update",
#             {
#                 "status": "IN_PROGRESS",
#                 "instruction": "✅ Action completed!",
#                 "face_aligned": True,
#             },
#         )
#         return

#     # ---------------- NORMAL ----------------
#     emit(
#         "server_update",
#         {
#             "status": "IN_PROGRESS",
#             "instruction": instruction_for(tracker.current_action),
#             "face_aligned": False,
#         },
#     )


# @socketio.on("liveness_finish")
# def ws_finish():
#     sid = request.sid
#     sess = SESSIONS.get(sid)

#     if not sess:
#         emit(
#             "server_update",
#             {
#                 "status": "FAILED",
#                 "error": "session_missing",
#             },
#         )
#         return

#     if sess.get("passed"):
#         emit(
#             "server_update",
#             {
#                 "status": "PASSED",
#                 "instruction": "✅ Already passed",
#             },
#         )
#     else:
#         status, meta = sess["tracker"].finalize()
#         emit(
#             "server_update",
#             {
#                 "status": "FAILED",
#                 "instruction": "❌ Liveness FAILED",
#                 "meta": meta,
#             },
#         )

#     SESSIONS.pop(sid, None)


# @socketio.on("disconnect")
# def ws_disconnect():
#     SESSIONS.pop(request.sid, None)

# # --------------------------------------------------
# # Run
# # --------------------------------------------------

# if __name__ == "__main__":
#     socketio.run(
#         app,
#         host="127.0.0.1",
#         port=PORT,
#         debug=False,
#     )









from flask import Flask, jsonify, send_from_directory, request
from flask_socketio import SocketIO, emit
import os
import time
import numpy as np
import cv2

from services.liveness_service import LivenessService, ActiveLivenessTracker
from services.spoof_service import SpoofDetector   # 🔐 SPOOF ADDITION

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)

PORT = int(os.getenv("FLASK_PORT", "5002"))

live_service = LivenessService()
spoof_model = SpoofDetector()   #  SPOOF ADDITION

SESSIONS = {}
TTL_SECONDS = 60


def instruction_for(action: str):
    if action == "LEFT": return "Turn your head LEFT 👈"
    if action == "RIGHT": return "Turn your head RIGHT 👉"
    if action == "BLINK": return "Blink your eyes 👁️"
    if action == "SMILE": return "Smile 🙂"
    if action == "NOD": return "Nod your head up & down ⬆️⬇️"
    return "Follow the instructions"


def decode_jpeg_to_bgr(jpeg_bytes: bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.get("/")
def index():
    return send_from_directory(".", "upload.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "port": PORT})


@socketio.on("connect")
def ws_connect():
    emit("server_update", {
        "status": "IN_PROGRESS",
        "action": "START",
        "instruction": "Connected ✅ Click Start"
    })


@socketio.on("liveness_start")
def ws_start():
    sid = request.sid
    tracker = ActiveLivenessTracker(live_service.face_mesh)

    SESSIONS[sid] = {
        "tracker": tracker,
        "frames_seen": 0,
        "last_seen": time.time(),
        "passed": False,
    }

    emit("server_update", {
        "status": "IN_PROGRESS",
        "action": tracker.current_action,
        "instruction": instruction_for(tracker.current_action),
    })


@socketio.on("liveness_frame")
def ws_frame(jpeg_bytes):

    sid = request.sid
    sess = SESSIONS.get(sid)
    if not sess:
        return

    now = time.time()

    # ---------------- GREEN BUFFER HOLD ----------------
    if sess.get("hold_green"):
        if now - sess["last_action_time"] < 3:
            emit("server_update", {
                "status": "IN_PROGRESS",
                "instruction": "✅ Action completed!",
                "face_aligned": True
            })
            return
        else:
            sess["hold_green"] = False
            tracker = sess["tracker"]

            if tracker.current_action is None:
                emit("server_update", {
                    "status": "PASSED",
                    "instruction": "✅ Liveness PASSED Successfully",
                    "face_aligned": True
                })
                SESSIONS.pop(sid, None)
                return

            emit("server_update", {
                "status": "IN_PROGRESS",
                "instruction": instruction_for(tracker.current_action),
                "face_aligned": False
            })
            return

    # ---------------- FRAME DECODE ----------------
    if isinstance(jpeg_bytes, bytearray):
        jpeg_bytes = bytes(jpeg_bytes)

    frame = decode_jpeg_to_bgr(jpeg_bytes)
    if frame is None:
        return

    sess["frames_seen"] += 1
    tracker = sess["tracker"]

    status, meta = tracker.update(frame, sess["frames_seen"])

    # 🔐 SPOOF ADDITION (NON-INTRUSIVE)
    landmarks = meta.get("landmarks")
    if landmarks and sess["frames_seen"] % 3 == 0:
        is_spoof, score = spoof_model.predict(landmarks)
        if is_spoof:
            emit("server_update", {
                "status": "FAILED",
                "instruction": "❌ Spoof detected",
                "reason": "SPOOF_DETECTED"
            })
            SESSIONS.pop(sid, None)
            return

    # ---------------- FAILED ----------------
    if status == "FAILED":
        emit("server_update", {
            "status": "FAILED",
            "instruction": "❌ Time up! Action not performed",
            "reason": meta
        })
        SESSIONS.pop(sid, None)
        return

    # ---------------- PASSED ----------------
    if status == "PASSED":
        emit("server_update", {
            "status": "PASSED",
            "instruction": "✅ Liveness PASSED Successfully",
            "face_aligned": True
        })
        SESSIONS.pop(sid, None)
        return

    # ---------------- ACTION COMPLETED ----------------
    if meta.get("action_done"):
        sess["hold_green"] = True
        sess["last_action_time"] = now
        emit("server_update", {
            "status": "IN_PROGRESS",
            "instruction": "✅ Action completed!",
            "face_aligned": True
        })
        return

    # ---------------- NORMAL ----------------
    emit("server_update", {
        "status": "IN_PROGRESS",
        "instruction": instruction_for(tracker.current_action),
        "face_aligned": False
    })


@socketio.on("disconnect")
def ws_disconnect():
    SESSIONS.pop(request.sid, None)


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=PORT, debug=False)
