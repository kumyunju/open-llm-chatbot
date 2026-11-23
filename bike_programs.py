import json
import math
import re


LINEAR_DATA = [
    (0, 120), (2, 180), (4, 240), (6, 310), (8, 390),
    (10, 480), (12, 570), (14, 670), (16, 780), (18, 900),
    (20, 1030), (22, 1150), (24, 1260), (26, 1370), (28, 1480),
    (30, 1560), (32, 1630), (34, 1680),
]

QUADRATIC_COEFFICIENTS = (
    2.7880781499202385, -110.74772157666828, 1551.3961038960995
)


def predict_linear(temperature):
    mean_x = sum(x for x, _ in LINEAR_DATA) / len(LINEAR_DATA)
    mean_y = sum(y for _, y in LINEAR_DATA) / len(LINEAR_DATA)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in LINEAR_DATA)
    slope /= sum((x - mean_x) ** 2 for x, _ in LINEAR_DATA)
    intercept = mean_y - slope * mean_x
    prediction = intercept + slope * temperature
    return max(0, math.floor(prediction + 0.5))


def predict_quadratic(temperature):
    a, b, c = QUADRATIC_COEFFICIENTS
    return max(0, math.floor(a * temperature * temperature + b * temperature + c + 0.5))


def run_ann(temperature, humidity, wind_speed):
    from ann_model import predict_ann

    return predict_ann(temperature, humidity, wind_speed)


PROGRAMS = {
    "predict_linear_bikes": ("선형회귀", predict_linear, ("temperature",)),
    "predict_quadratic_bikes": ("비선형회귀", predict_quadratic, ("temperature",)),
    "predict_ann_bikes": ("인공신경망", run_ann, ("temperature", "humidity", "wind_speed")),
}

INPUTS = {
    "temperature": "기온(°C)",
    "humidity": "습도(%)",
    "wind_speed": "풍속(m/s)",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{label} 프로그램으로 자전거 대여량을 예측한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "number", "description": INPUTS[key]}
                    for key in fields
                },
                "required": list(fields),
            },
        },
    }
    for name, (label, _, fields) in PROGRAMS.items()
]

NUMBER = r"([+-]?\d+(?:\.\d+)?)"
PATTERNS = {
    "temperature": rf"(?:기온|온도)(?:은|이|가)?[:=]?{NUMBER}|{NUMBER}(?:도|℃|°c)",
    "humidity": rf"습도(?:는|가)?[:=]?{NUMBER}|{NUMBER}(?:%|퍼센트)",
    "wind_speed": rf"풍속(?:은|이)?[:=]?{NUMBER}|{NUMBER}(?:m/s|미터/초)|초속{NUMBER}",
}

MODEL_PATTERNS = {
    "predict_linear_bikes": r"(?<!비)선형회귀",
    "predict_quadratic_bikes": r"비선형|2차회귀|2차식|이차회귀|이차식",
    "predict_ann_bikes": r"신경망|(?<![a-z])(?:ann|mlp)(?![a-z])",
}


def read_values(text):
    compact = "".join(text.lower().split())
    values = {}

    for key, pattern in PATTERNS.items():
        matches = list(re.finditer(pattern, compact))
        if len(matches) == 1:
            number = next(value for value in matches[0].groups() if value is not None)
            values[key] = float(number)

    return values


def run_prediction(name, values):
    label, predict, fields = PROGRAMS[name]
    missing = [INPUTS[key] for key in fields if key not in values]
    if missing:
        return f"{label} 예측에 필요한 값을 알려주세요. {', '.join(missing)}"

    try:
        args = [values[key] for key in fields]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in args):
            return "예측 입력값을 숫자로 알려주세요."
        count = predict(*args)
    except FileNotFoundError:
        return "저장소에서 SeoulBikeData_1000_3inputs.csv 파일을 확인해 주세요."
    except ModuleNotFoundError:
        return "requirements.txt에 pandas와 scikit-learn이 있는지 확인해 주세요."
    except ValueError as error:
        return str(error) or "예측 입력값을 확인해 주세요."
    except OverflowError:
        return "예측 입력값을 확인해 주세요."

    inputs = " · ".join(f"{INPUTS[key]} {values[key]:g}" for key in fields)
    return f"{label} 프로그램의 예측값은 약 **{count:,}대**예요.\n\n{inputs}"


def answer_chat(client, history, user_entry):
    text = user_entry["content"]
    compact = "".join(text.lower().split())
    matches = [
        name for name, pattern in MODEL_PATTERNS.items()
        if re.search(pattern, compact)
    ]
    values = read_values(text)

    if not re.search(
        r"원리|방법|차이|왜|회귀식|기록|지난번|설명|날씨|건조|따뜻|춥|더워", compact
    ):
        if len(matches) == 1 and re.search(r"예측|몇대|대여량|계산|돌려", compact):
            return run_prediction(matches[0], values)

        if not matches and values and history:
            for name, (label, _, _) in PROGRAMS.items():
                prefix = f"{label} 예측에 필요한 값을 알려주세요."
                if not history[-1]["content"].startswith(prefix):
                    continue

                for entry in reversed(history):
                    if entry["role"] == "assistant" and not entry["content"].startswith(prefix):
                        break
                    if entry["role"] == "user":
                        values = {**read_values(entry["content"]), **values}

                return run_prediction(name, values)

    system = (
        "한국어로 자연스럽게 대화하세요. 기존 프로그램으로 자전거 대여량 예측을 요청하면 "
        "지정한 모델의 도구를 사용하세요. 모델이나 입력값이 없으면 질문하세요. "
        "선형·비선형회귀는 기온, 인공신경망은 기온·습도·풍속이 필요해요."
    )

    result = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "system", "content": system}, *history, user_entry],
        tools=TOOLS,
        tool_choice="auto",
    )

    reply = result.choices[0].message
    if not reply.tool_calls:
        return reply.content

    for call in reply.tool_calls:
        if call.function.name in PROGRAMS:
            try:
                args = json.loads(call.function.arguments)
            except (ValueError, TypeError):
                args = {}

            return run_prediction(call.function.name, args if isinstance(args, dict) else {})

    return "사용할 예측 프로그램과 조건을 알려주세요."
