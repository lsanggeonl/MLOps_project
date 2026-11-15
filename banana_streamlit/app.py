# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import uuid
import io
import time


# ============================================================
# 1. Streamlit 설정
# ============================================================

st.set_page_config(
    page_title="Banana AI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. 경로
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

STORAGE_DIR = BASE_DIR / "storage"
ANALYSIS_DIR = STORAGE_DIR / "analyses"

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 클래스
# ============================================================

# 반드시 실제 best.pt의 class 순서와 맞춰야 함
CLASS_NAMES = {
    0: "unripe",
    1: "ripe",
    2: "overripe",
    3: "rotten",
}


DISPLAY_NAMES = {
    "unripe": "미숙",
    "ripe": "적숙",
    "overripe": "과숙",
    "rotten": "부패",
}


CLASS_EMOJI = {
    "unripe": "🟢",
    "ripe": "🟡",
    "overripe": "🟠",
    "rotten": "🔴",
}


# ============================================================
# 4. 숙성도별 사용자 콘텐츠
# ============================================================

RIPENESS_INFO = {

    "unripe": {

        "title": "조금 더 숙성시켜 보세요!",

        "description":
            "아직 충분히 숙성되지 않은 상태입니다. "
            "단맛이 상대적으로 부족할 수 있습니다.",

        "cooking": [
            "팬에 구워 먹기",
            "튀김이나 구이 요리에 활용하기",
            "조금 더 숙성시킨 뒤 그대로 먹기",
        ],

        "eating": [
            "실온에 두고 조금 더 숙성시키기",
            "껍질이 노랗게 변할 때까지 기다리기",
        ],

        "facts": [
            "바나나는 수확 후에도 숙성이 진행되는 후숙 과일입니다.",
            "숙성이 진행되면서 전분이 당으로 전환되어 단맛이 증가합니다.",
        ],
    },


    "ripe": {

        "title": "지금이 맛있게 먹기 좋은 상태예요! 😋",

        "description":
            "단맛과 식감의 균형이 좋은 상태로 판단됩니다.",

        "cooking": [
            "바나나 우유",
            "바나나 스무디",
            "팬케이크 토핑",
            "요거트 토핑",
        ],

        "eating": [
            "그대로 먹기",
            "요거트와 함께 먹기",
            "시리얼이나 오트밀에 곁들이기",
        ],

        "facts": [
            "바나나는 숙성이 진행될수록 일반적으로 단맛이 강해집니다.",
            "숙성 과정에서 엽록소가 감소하면서 껍질이 노란색으로 변합니다.",
        ],
    },


    "overripe": {

        "title": "베이킹이나 가공 요리에 활용해보세요! 🍳",

        "description":
            "과숙 상태입니다. 과육이 부드럽고 단맛이 강할 수 있어 "
            "베이킹이나 디저트에 활용하기 좋습니다.",

        "cooking": [
            "바나나 브레드",
            "바나나 팬케이크",
            "바나나 스무디",
            "바나나잼",
        ],

        "eating": [
            "베이킹 반죽에 으깨서 넣기",
            "스무디에 넣어 자연스러운 단맛 더하기",
            "냉동 후 디저트처럼 활용하기",
        ],

        "facts": [
            "과숙 바나나는 부드러운 식감 때문에 베이킹 재료로 활용하기 좋습니다.",
            "숙성이 진행되면 당도가 높아져 디저트 재료로 활용하기 좋습니다.",
        ],
    },


    "rotten": {

        "title": "섭취하지 않는 것을 권장해요. ⚠️",

        "description":
            "모델이 부패 상태로 판단했습니다. "
            "실제 섭취 여부는 곰팡이, 냄새 등 실제 상태도 확인해야 합니다.",

        "cooking": [
            "섭취용 조리는 권장하지 않습니다.",
        ],

        "eating": [
            "곰팡이가 보이면 폐기하세요.",
            "이상한 냄새가 난다면 섭취하지 마세요.",
        ],

        "facts": [
            "AI의 분류 결과만으로 식품의 안전성을 확정할 수는 없습니다.",
            "부패가 의심되는 식품은 섭취하지 않는 것이 안전합니다.",
        ],
    },
}


# ============================================================
# 5. 모델 로드
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None

    return YOLO(
        str(MODEL_PATH)
    )


model = load_model()


# ============================================================
# 6. 유틸
# ============================================================

def get_now():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def create_analysis_folder():

    analysis_id = uuid.uuid4().hex[:12]

    folder = (
        ANALYSIS_DIR /
        analysis_id
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return analysis_id, folder


# ============================================================
# 7. YOLO 추론
# ============================================================

def run_yolo(image):

    results = model(
        image,
        verbose=False
    )

    result = results[0]

    detections = []


    if result.boxes is None:

        return result, detections


    for i in range(
        len(result.boxes)
    ):

        class_id = int(
            result.boxes.cls[i].item()
        )

        confidence = float(
            result.boxes.conf[i].item()
        )


        # 프로젝트에서 지정한 클래스
        class_name = CLASS_NAMES.get(
            class_id
        )


        # 혹시 mapping에 없는 class라면
        # YOLO 자체 names 사용
        if class_name is None:

            class_name = str(
                result.names.get(
                    class_id,
                    class_id
                )
            ).lower()


        bbox = (
            result
            .boxes
            .xyxy[i]
            .tolist()
        )


        detections.append({

            "object_id":
                i + 1,

            "class_id":
                class_id,

            "class_name":
                class_name,

            "confidence":
                round(
                    confidence,
                    6
                ),

            "bbox": [
                round(
                    float(x),
                    2
                )
                for x in bbox
            ]
        })


    return result, detections


# ============================================================
# 8. 대표 숙성도
# ============================================================

def get_representative_result(
    detections
):

    if not detections:

        return None


    # 가장 많이 검출된 숙성도를 대표 결과로 사용
    counts = {}


    for detection in detections:

        name = detection[
            "class_name"
        ]

        counts[name] = (
            counts.get(
                name,
                0
            ) + 1
        )


    return max(
        counts,
        key=counts.get
    )


# ============================================================
# 9. 분석 결과 저장
# ============================================================

def save_analysis(
    uploaded_file,
    original_image,
    result_image,
    detections
):

    analysis_id, folder = (
        create_analysis_folder()
    )


    original_path = (
        folder /
        "original.jpg"
    )

    result_path = (
        folder /
        "result.jpg"
    )

    json_path = (
        folder /
        "result.json"
    )


    # 원본 저장
    original_image.save(
        original_path,
        format="JPEG",
        quality=95
    )


    # 결과 이미지 저장
    result_image.save(
        result_path,
        format="JPEG",
        quality=95
    )


    representative = (
        get_representative_result(
            detections
        )
    )


    recommendation = None


    if representative:

        recommendation = (
            RIPENESS_INFO.get(
                representative
            )
        )


    data = {

        "analysis_id":
            analysis_id,

        "created_at":
            get_now(),

        "original_filename":
            uploaded_file.name,

        "model":
            MODEL_PATH.name,

        "representative_result":
            representative,

        "representative_display":
            DISPLAY_NAMES.get(
                representative,
                representative
            )
            if representative
            else None,

        "detections":
            detections,

        "recommendation":
            recommendation
    }


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


    return data


# ============================================================
# 10. 기록 조회
# ============================================================

def get_history():

    history = []


    for folder in ANALYSIS_DIR.iterdir():

        if not folder.is_dir():

            continue


        json_path = (
            folder /
            "result.json"
        )


        if not json_path.exists():

            continue


        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

                history.append(data)

        except Exception:

            pass


    history.sort(
        key=lambda x:
        x.get(
            "created_at",
            ""
        ),
        reverse=True
    )


    return history


# ============================================================
# 11. 기록 하나 가져오기
# ============================================================

def get_history_item(
    analysis_id
):

    folder = (
        ANALYSIS_DIR /
        analysis_id
    )


    json_path = (
        folder /
        "result.json"
    )


    if not json_path.exists():

        return None


    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# 12. 기록 삭제
# ============================================================

def delete_history(
    analysis_id
):

    folder = (
        ANALYSIS_DIR /
        analysis_id
    )


    if not folder.exists():

        return


    for file in folder.iterdir():

        if file.is_file():

            file.unlink()


    folder.rmdir()


# ============================================================
# 13. 결과 화면
# ============================================================

def show_result(
    data
):

    analysis_id = data[
        "analysis_id"
    ]


    folder = (
        ANALYSIS_DIR /
        analysis_id
    )


    original_path = (
        folder /
        "original.jpg"
    )

    result_path = (
        folder /
        "result.jpg"
    )


    # --------------------------------------------------------
    # 결과 이미지
    # --------------------------------------------------------

    st.subheader(
        "📊 AI 분석 결과"
    )


    _, col1, col2, _ = st.columns(
        [1, 1, 1, 1]
    )


    with col1:

        st.markdown(
            "### 📷 원본"
        )

        if original_path.exists():

            st.image(
                str(original_path),
                use_container_width=True
            )


    with col2:

        st.markdown(
            "### 🤖 AI 분석"
        )

        if result_path.exists():

            st.image(
                str(result_path),
                use_container_width=True
            )


    # --------------------------------------------------------
    # 객체별 결과
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🍌 객체별 숙성도"
    )


    detections = data[
        "detections"
    ]


    if not detections:

        st.warning(
            "바나나 객체를 찾지 못했습니다."
        )

        return


    columns = st.columns(
        min(
            3,
            len(detections)
        )
    )


    for index, detection in enumerate(
        detections
    ):

        class_name = detection[
            "class_name"
        ]

        display_name = (
            DISPLAY_NAMES.get(
                class_name,
                class_name
            )
        )


        emoji = (
            CLASS_EMOJI.get(
                class_name,
                "🍌"
            )
        )


        confidence = detection[
            "confidence"
        ]


        with columns[
            index %
            len(columns)
        ]:

            st.markdown(
                f"### {emoji} 객체 "
                f"{detection['object_id']}"
            )


            st.markdown(
                f"## {display_name}"
            )


            st.progress(
                confidence
            )


            st.write(
                f"AI 신뢰도: "
                f"**{confidence:.2%}**"
            )


    # --------------------------------------------------------
    # 대표 결과
    # --------------------------------------------------------

    representative = data.get(
        "representative_result"
    )


    if representative:

        st.divider()

        info = RIPENESS_INFO.get(
            representative
        )


        st.subheader(
            f"{CLASS_EMOJI.get(representative, '🍌')} "
            f"전체 대표 결과"
        )


        st.markdown(
            f"## "
            f"{DISPLAY_NAMES.get(representative, representative)}"
        )


        if info:

            st.write(
                info["description"]
            )


    # --------------------------------------------------------
    # 추천 콘텐츠
    # --------------------------------------------------------

    recommendation = data.get(
        "recommendation"
    )


    if recommendation:

        st.divider()


        st.subheader(
            f"🍳 {recommendation['title']}"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                "### 🍳 추천 조리 방법"
            )


            for item in (
                recommendation["cooking"]
            ):

                st.write(
                    f"• {item}"
                )


        with col2:

            st.markdown(
                "### 😋 맛있게 먹는 방법"
            )


            for item in (
                recommendation["eating"]
            ):

                st.write(
                    f"• {item}"
                )


        st.markdown(
            "### 💡 바나나 상식"
        )


        for fact in (
            recommendation["facts"]
        ):

            st.info(
                fact
            )


    st.caption(
        f"분석 시간: "
        f"{data['created_at']}"
    )


# ============================================================
# 14. 세션 상태
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = (
        "new"
    )


if "selected_history" not in st.session_state:

    st.session_state.selected_history = None


if "delete_target" not in st.session_state:

    st.session_state.delete_target = None


# ============================================================
# 15. Sidebar
# ============================================================

with st.sidebar:

    st.title(
        "🍌 Banana AI"
    )


    st.caption(
        "AI 기반 바나나 숙성도 판별"
    )


    st.divider()


    # 새 분석
    if st.button(
        "＋ 새 분석",
        use_container_width=True
    ):

        st.session_state.page = (
            "new"
        )

        st.session_state.selected_history = (
            None
        )

        st.rerun()


    st.markdown(
        "### 🕘 이전 추론 기록"
    )


    history = get_history()


    if not history:

        st.caption(
            "아직 분석 기록이 없습니다."
        )


    for item in history:

        analysis_id = item[
            "analysis_id"
        ]


        created_at = item[
            "created_at"
        ]


        result_name = item.get(
            "representative_display",
            "미검출"
        )


        # 기록 선택
        col1, col2 = st.columns(
            [5, 1]
        )


        with col1:

            if st.button(
                f"{result_name} · "
                f"{created_at[5:16]}",
                key=(
                    f"history_"
                    f"{analysis_id}"
                ),
                use_container_width=True
            ):

                st.session_state.page = (
                    "history"
                )

                st.session_state.selected_history = (
                    analysis_id
                )

                st.rerun()


        # 쓰레기통
        with col2:

            if st.button(
                "🗑️",
                key=(
                    f"delete_"
                    f"{analysis_id}"
                ),
                help="기록 삭제"
            ):

                st.session_state.delete_target = (
                    analysis_id
                )

                st.rerun()


    st.divider()


    if st.button(
        "ℹ️ 모델 정보",
        use_container_width=True
    ):

        st.session_state.page = (
            "model"
        )

        st.session_state.selected_history = (
            None
        )

        st.rerun()


# ============================================================
# 16. 삭제 확인
# ============================================================

if st.session_state.delete_target:

    target = (
        st.session_state.delete_target
    )


    st.warning(
        "이 분석 기록을 삭제하시겠습니까?"
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "삭제",
            type="primary"
        ):

            delete_history(
                target
            )


            st.session_state.delete_target = (
                None
            )


            if (
                st.session_state.selected_history
                == target
            ):

                st.session_state.selected_history = (
                    None
                )


            st.rerun()


    with col2:

        if st.button(
            "취소"
        ):

            st.session_state.delete_target = (
                None
            )

            st.rerun()


# ============================================================
# 17. 새 분석 화면
# ============================================================

elif st.session_state.page == "new":

    st.title(
        "🍌 바나나 숙성도 AI"
    )


    st.write(
        "사진을 업로드하면 "
        "YOLO11n Bounding Box 모델을 이용하여 "
        "바나나의 숙성도를 분석합니다."
    )


    st.divider()


    # 모델 존재 확인
    if model is None:

        st.error(
            "YOLO 모델을 찾을 수 없습니다."
        )


        st.code(
            "models/best.pt"
        )


        st.stop()


    # 이미지 업로드
    uploaded_file = st.file_uploader(
        "📷 바나나 사진 업로드",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file:

        image = Image.open(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        ).convert("RGB")


        st.image(
            image,
            caption="업로드된 이미지",
            use_container_width=True
        )


        st.divider()


        if st.button(
            "🤖 숙성도 분석 시작",
            type="primary",
            use_container_width=True
        ):


            # ------------------------------------------------
            # 분석 진행 표시
            # ------------------------------------------------

            with st.status(
                "🤖 AI가 분석하고 있습니다...",
                expanded=True
            ) as status:


                st.write(
                    "📷 이미지 확인 중..."
                )

                time.sleep(
                    0.2
                )


                st.write(
                    "⚙️ 이미지 전처리 중..."
                )

                time.sleep(
                    0.2
                )


                st.write(
                    "🔍 바나나 객체 탐색 중..."
                )


                start_time = time.perf_counter()


                result, detections = (
                    run_yolo(
                        image
                    )
                )


                elapsed = (
                    time.perf_counter()
                    - start_time
                )


                st.write(
                    f"🧠 숙성도 분석 완료 "
                    f"({elapsed:.2f}초)"
                )


                st.write(
                    "📊 결과 정리 중..."
                )


                # YOLO 결과 이미지
                plotted = (
                    result.plot()
                )


                # YOLO plot은 BGR
                result_image = (
                    Image.fromarray(
                        plotted[..., ::-1]
                    )
                )


                st.write(
                    "💾 분석 기록 저장 중..."
                )


                data = save_analysis(
                    uploaded_file,
                    image,
                    result_image,
                    detections
                )


                status.update(
                    label="✅ 분석 완료!",
                    state="complete"
                )


            # 분석 결과 표시
            show_result(
                data
            )


# ============================================================
# 18. 이전 기록 화면
# ============================================================

elif st.session_state.page == "history":

    analysis_id = (
        st.session_state.selected_history
    )


    if not analysis_id:

        st.info(
            "왼쪽에서 이전 분석 기록을 선택하세요."
        )


    else:

        data = get_history_item(
            analysis_id
        )


        if data is None:

            st.error(
                "분석 기록을 찾을 수 없습니다."
            )


        else:

            st.title(
                "🕘 이전 분석 결과"
            )


            st.caption(
                f"{data['created_at']} · "
                f"{data['original_filename']}"
            )


            show_result(
                data
            )


# ============================================================
# 19. 모델 정보
# ============================================================

elif st.session_state.page == "model":

    st.title(
        "ℹ️ 모델 정보"
    )


    st.markdown(
        "### YOLO 모델"
    )


    st.write(
        "YOLO11n Bounding Box"
    )


    st.markdown(
        "### Weight"
    )


    st.code(
        str(MODEL_PATH)
    )


    st.markdown(
        "### 클래스"
    )


    for class_id, class_name in (
        CLASS_NAMES.items()
    ):

        st.write(
            f"{class_id} : "
            f"{DISPLAY_NAMES.get(class_name, class_name)}"
        )


    st.info(
        "현재 서비스 추론은 Bounding Box 모델만 "
        "사용합니다. Segmentation 모델과의 성능 비교는 "
        "발표/평가 내용으로 별도 구성합니다."
    )