import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def draw_face_mesh(frame, faceLandmarks):

    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=faceLandmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=faceLandmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=faceLandmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
    )


def capture_frame(frame_path="temp.png"):
    """
     function to capture webcam feed, detect face landmarks, and save images.
    """
    videoCapture = cv2.VideoCapture(-1)

    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frameRate = 30

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as faceMesh:
        while videoCapture.isOpened():
            isSuccess, frame = videoCapture.read()
            if not isSuccess:
                break
            originalFrame = frame.copy()
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceMesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for faceLandmarks in results.multi_face_landmarks:
                    draw_face_mesh(frame, faceLandmarks)

            cv2.imshow('Web cam face mesh', cv2.flip(frame, 1))

            key = cv2.waitKey(int(1000 / frameRate)) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('s'):  # 's' key
                cv2.imwrite(frame_path, originalFrame)
                break

        videoCapture.release()
        cv2.destroyAllWindows()
