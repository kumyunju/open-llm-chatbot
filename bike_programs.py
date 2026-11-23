import json
import math


LINEAR_DATA = [
    (0, 120), (2, 180), (4, 240), (6, 310), (8, 390),
    (10, 480), (12, 570), (14, 670), (16, 780), (18, 900),
    (20, 1030), (22, 1150), (24, 1260), (26, 1370), (28, 1480),
    (30, 1560), (32, 1630), (34, 1680),
]

# 기존 비선형회귀 원본에서 계산한 2차식의 a, b, c
QUADRATIC_COEFFICIENTS = (
    2.7880781499202385,
    -110.74772157666828,
    1551.3961038960995,
)


def temperature_tool(name, description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number", "description": "섭씨 기온"},
                },
                "required": ["temperature"],
            },
        },
    }


TOOLS = [
    temperature_tool(
        "predict_linear_bikes", "선형회귀 직선식으로 자전거 대여량을 예측한다."
    ),
    temperature_tool(
        "predict_quadratic_bikes", "비선형회귀의 U자형 2차식으로 자전거 대여량을 예측한다."
    ),
]


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
    prediction = a * temperature * temperature + b * temperature + c
    return max(0, math.floor(prediction + 0.5))


PREDICTORS = {
    "predict_linear_bikes": ("선형회귀", predict_linear),
    "predict_quadratic_bikes": ("비선형회귀", predict_quadratic),
}


def answer_chat(client, history, user_entry):
    system = (
        "한국어로 자연스럽게 대화해 주세요. "
        "사용자가 기존 프로그램으로 자전거 대여량 예측을 요청하면 해당 도구를 사용하세요. "
        "선형회귀는 predict_linear_bikes, 비선형회귀 또는 2차 회귀는 predict_quadratic_bikes예요. "
        "사용자가 모델을 정하지 않았다면 어느 프로그램으로 예측할지 물어보세요. "
        "기온 정보가 대화에 없으면 기온을 물어보세요. "
        "실행한 프로그램의 결과는 도구를 사용해서 확인하세요."
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
        program = PREDICTORS.get(call.function.name)
        if program is None:
            continue

        label, predict = program
        try:
            temperature = json.loads(call.function.arguments)["temperature"]
            if type(temperature) not in (int, float) or not math.isfinite(temperature):
                raise ValueError
            count = predict(temperature)
        except (ValueError, KeyError, TypeError, OverflowError):
            return "예측할 섭씨 기온을 숫자로 알려주세요."

        return f"{label} 프로그램으로 예측한 {temperature:g}°C의 자전거 대여량은 약 **{count:,}대**예요."

    return "어떤 예측을 원하는지 다시 알려주세요."
