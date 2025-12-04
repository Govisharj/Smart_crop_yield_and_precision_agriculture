import requests

# Put your actual API key here
API_KEY = '6b25fff73156bcfbd67f1115e96d2251'  # OpenWeatherMap API Key


def get_weather_forecast(location: str):
    """
    Fetch today's and tomorrow's rain probability using OpenWeatherMap API.

    :param location: City,CountryCode (e.g., "Chennai,IN")
    :return: (today_rain, tomorrow_rain) in percentages
    """
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if "list" not in data:
        raise ValueError(f"Error fetching weather: {data.get('message', 'Unknown error')}")

    forecast_list = data["list"]

    def precip_probability(item):
        # Base on PoP if present
        prob = float(item.get("pop", 0) or 0)

        # Consider rain volume if present (scale conservatively: 0.0-1.0 at 5mm/3h)
        rain = item.get("rain") or {}
        if isinstance(rain, dict):
            vol = float(rain.get("3h") or rain.get("1h") or 0)
            if vol > 0:
                vol_prob = min(vol / 5.0, 1.0)
                prob = max(prob, vol_prob)

        # If weather condition indicates rain-like phenomena, set a floor (but not 100%)
        weather_arr = item.get("weather") or []
        if weather_arr:
            main = (weather_arr[0] or {}).get("main", "")
            if main in {"Rain", "Drizzle", "Thunderstorm"}:
                prob = max(prob, 0.5)

        return max(0.0, min(prob, 1.0))

    # First 8 forecasts: Next 24 hours, Next 8: Following 24 hours (3h step)
    today_slice = forecast_list[:8]
    tomorrow_slice = forecast_list[8:16] if len(forecast_list) >= 16 else forecast_list[8:]

    def combined_probability(slice_items):
        # Probability of at least one rainy slot in the slice: 1 - Π(1 - p_i)
        no_rain_prob = 1.0
        has_item = False
        for it in slice_items:
            p = precip_probability(it)
            has_item = True
            no_rain_prob *= (1.0 - p)
        if not has_item:
            return 0.0
        return 1.0 - no_rain_prob

    today_prob = combined_probability(today_slice)
    tomorrow_prob = combined_probability(tomorrow_slice)

    return round(today_prob * 100, 2), round(tomorrow_prob * 100, 2)


