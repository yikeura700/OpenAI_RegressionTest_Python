from openai import OpenAI
import pandas as pd
import numpy as np
import json

# Import OpenAI API key and endpoint from config file
with open("config.json", "r") as f:
    config = json.load(f)

client = OpenAI(
    api_key=config["api_key"],
    base_url=config.get("endpoint", "https://api.openai.com/v1")
)

# Define the model version
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Setting the threshold for cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Execute regression test
def regression_test(prompt, expected_output, threshold=0.9):
    response = client.chat.completions.create(
        model=config["model_version"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    current_output = response.choices[0].message.content.strip()

    expected_embedding = get_embedding(expected_output)
    current_embedding = get_embedding(current_output)

    similarity = cosine_similarity(expected_embedding, current_embedding)
    result = "PASS" if similarity >= threshold else "FAIL"

    return {
        "Prompt": prompt,
        "ExpectedOutput": expected_output,
        "CurrentOutput": current_output,
        "Similarity": similarity,
        "Result": result
    }

# Import test cases from Excel
df = pd.read_excel("test_cases.xlsx")

# Execute regression tests
results = []
for _, row in df.iterrows():
    result = regression_test(row["Prompt"], row["ExpectedOutput"])
    results.append(result)

# Save results to Excel
results_df = pd.DataFrame(results)
results_df.to_excel("regression_test_results.xlsx", index=False)

print("The test finished：The result saved as regression_test_results.xlsx")
