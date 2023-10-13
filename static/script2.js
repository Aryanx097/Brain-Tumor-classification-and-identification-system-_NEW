// Simulate result data (replace with actual data from your ML model)
const result = {
    prediction: "Brain Tumor Detected",
    probability: "80%"
};

// Update the page with the result
document.getElementById("prediction").textContent = result.prediction;
document.getElementById("probability").textContent = result.probability;
