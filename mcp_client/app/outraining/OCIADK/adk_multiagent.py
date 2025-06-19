import oci,os, re
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any
from oci.addons.adk import Agent, AgentClient, tool

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file(profile_name=CONFIG_PROFILE)  # Update this with your own profile name
sess_id = ""

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
print(PROJECT_ROOT)
# Set up the OCI GenAI Agents endpoint configuration
AGENT_EP_ID = os.getenv("AGENT_EP_ID")
AGENT_EP_ID_2 = "ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaawe6j4fqa4hiv7nfbfmp65gwcuxbuncjovhtzx74rjfvbyedqxf6q"
AGENT_SERVICE_EP = os.getenv("AGENT_SERVICE_EP")


# === Custom Function for City Name Extraction from user query ===
def extract_city(input_text: str) -> str:
    # List of possible cities
    cities = ["Seattle", "New York", "Chicago"]

    # Create a pattern for case-insensitive search
    pattern = re.compile(r'\b(' + '|'.join(re.escape(city) for city in cities) + r')\b', re.IGNORECASE)

    # Search for the city in the input string
    match = pattern.search(input_text)

    if match:
        # Return the correctly capitalized city from the original list
        matched_city = match.group(0).lower()
        for city in cities:
            if city.lower() == matched_city:
                return city
    return None


# === Custom Message class ===
class Message(dict):
    def __init__(self, role: str, content: str):
        super().__init__(role=role, content=content)
        self.role = role
        self.content = content


# === Tool for WeatherAgent ===
@tool
def get_weather(location: str) -> Dict[str, str]:
    """
    Get the weather for a given location.
    """

    facts = {
        "Seattle": "72",
        "New York": "82",
        "Chicago": "92"
    }
    fact = facts.get(location, "Sorry, I don't have a temp for that city.")
    return {"location": location, "temperature": fact, "unit": "F"}


# === Tool for CityFactAgent ===
@tool
def get_city_fact(city: str) -> Dict[str, str]:
    """
    Get a fun fact about the city.
    """
    facts = {
        "Seattle": "Seattle is known as the Emerald City.",
        "New York": "New York City is home to the Statue of Liberty.",
        "Chicago": "Chicago is famous for its architecture and deep-dish pizza."
    }
    fact = facts.get(city, "Sorry, I don't have a fact for that city.")
    return {"city": city, "fact": fact}


# === Build individual agents ===
# Agentname : OCI-DEMO-AGENT-1
# Endpoint Name: genaiagentendpoint20250613083947
def build_weather_agent(client: AgentClient) -> Agent:
    return Agent(
        client=client,
        agent_endpoint_id=AGENT_EP_ID,
        instructions="You are a weather expert. Answer only weather-related questions.",
        tools=[get_weather]
    )

    # Agentname : OCI-DEMO-AGENT-2
    # Endpoint Name: genaiagentendpoint20250613084006


def build_city_fact_agent(client: AgentClient) -> Agent:
    return Agent(
        client=client,
        agent_endpoint_id=AGENT_EP_ID_2,
        instructions="You are a travel guide. Share interesting facts about cities.",
        tools=[get_city_fact]
    )


# === Main Coordinator Logic ===
def main():
    # --- Step 1: Setup client ---
    client = AgentClient(
        auth_type="api_key",
        profile=CONFIG_PROFILE,
        region="us-chicago-1",
    )

    # --- Step 2: Setup agents ---
    weather_agent = build_weather_agent(client)
    city_fact_agent = build_city_fact_agent(client)
    weather_agent.setup()
    city_fact_agent.setup()

    # --- Step 3: Take user input ---
    user_query = input("You: ")

    # --- Step 4: Coordinator logic to route query ---
    messages = [Message("user", user_query)]

    # get city from user_query
    city = extract_city(user_query)

    # setup rules to call agent and pass query task to them
    use_weather = "weather" in user_query.lower() or "temperature" in user_query.lower()
    use_fact = "fact" in user_query.lower() or "interesting" or "facts" in user_query.lower()

    results = []

    if use_weather:  # OCI-DEMO-AGENT-1
        response = weather_agent.run(input=f"What is the weather in {city}?", messages=messages)
        text = response.data["message"]["content"]["text"]
        results.append(text)

    if use_fact:  # OCI-DEMO-AGENT-2
        response = city_fact_agent.run(input=f"Tell me a fact about {city}.", messages=messages)
        text = response.data["message"]["content"]["text"]
        results.append(text)

    # --- Step 5: Combine results ---
    print("\nAssistant:")
    for result in results:
        print("- " + result)


if __name__ == "__main__":
    main()