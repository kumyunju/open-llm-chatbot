import json
import math


LINEAR_DATA = [
    (0, 120), (2, 180), (4, 240), (6, 310), (8, 390),
    (10, 480), (12, 570), (14, 670), (16, 780), (18, 900),
    (20, 1030), (22, 1150), (24, 1260), (26, 1370), (28, 1480),
    (30, 1560), (32, 1630), (34, 1680),
]

LINEAR_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_linear_bikes",
        "description": "기온으로 기존 선형회귀 프로그램의 자전거 대여량을 예측한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "description": "섭씨 기온"},
            },
            "required": ["temperature"],
        },
    },
}


def predict_linear(temperature):
    mean_x = sum(x for x, _ in LINEAR_DATA) / len(LINEAR_DATA)
    mean_y = sum(y for _, y in LINEAR_DATA) / len(LINEAR_DATA)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in LINEAR_DATA)
    slope /= sum((x - mean_x) ** 2 for x, _ in LINEAR_DATA)
    intercept = mean_y - slope * mean_x
    prediction = intercept + slope * temperature
    return max(0, math.floor(prediction + 0.5))


def answer_chat(client, history, user_entry):
    system = (
        "한국어로 자연스럽게 대화해 주세요. "
        "사용자가 기존 선형회귀 프로그램으로 자전거 대여량을 예측해 달라고 하면 "
        "predict_linear_bikes 도구를 사용하세요. "
        "기온 정보가 대화에 없으면 기온을 물어보세요. "
        "실행한 프로그램의 결과는 도구를 사용해서 확인하세요."
    )

    result = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "system", "content": system}, *history, user_entry],
        tools=[LINEAR_TOOL],
        tool_choice="auto",
    )

    reply = result.choices[0].message
    if not reply.tool_calls:
        return reply.content

    for call in reply.tool_calls:
        if call.function.name != "predict_linear_bikes":
            continue
        try:
            temperature = json.loads(call.function.arguments)["temperature"]
            if type(temperature) not in (int, float) or not math.isfinite(temperature):
                raise ValueError
            count = predict_linear(temperature)
        except (ValueError, KeyError, TypeError, OverflowError):
            return "예측할 섭씨 기온을 숫자로 알려주세요."

        return f"선형회귀 프로그램으로 예측한 {temperature:g}°C의 자전거 대여량은 약 **{count:,}대**예요."

    return "어떤 예측을 원하는지 다시 알려주세요."
