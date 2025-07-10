document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const resultIndicator = document.getElementById('result-indicator');
    const resultText = document.getElementById('result-text');
    const probabilityText = document.getElementById('probability-text');
    const messageText = document.getElementById('message-text');
    const resetBtn = document.getElementById('reset-btn');
    const loader = document.getElementById('loader');

    // API endpoint
    const API_URL = 'http://127.0.0.1:5000/predict'; // Make sure this is the correct endpoint

    // Form submission handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loader
        loader.classList.remove('hidden');

        // Get form values
        const features = [
            parseFloat(document.getElementById('pregnancies').value),
            parseFloat(document.getElementById('glucose').value),
            parseFloat(document.getElementById('bloodPressure').value),
            parseFloat(document.getElementById('skinThickness').value),
            parseFloat(document.getElementById('insulin').value),
            parseFloat(document.getElementById('bmi').value),
            parseFloat(document.getElementById('diabetesPedigree').value),
            parseFloat(document.getElementById('age').value)
        ];

        try {
            // Send prediction request
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ features })
            });

            // Hide loader
            loader.classList.add('hidden');

            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }

            const data = await response.json();

            // Format probability as percentage
            const probability = (data.probability * 100).toFixed(1);

            // Determine risk level based on probability
            let riskLevel, indicatorClass, indicatorSymbol, message;

            if (data.probability < 0.3) {
                riskLevel = "Low Risk";
                indicatorClass = "result-indicator negative";
                indicatorSymbol = "✓";
                message = "Based on the provided parameters, the risk of diabetes appears to be low. However, maintaining a healthy lifestyle is always recommended.";
            } else if (data.probability < 0.7) {
                riskLevel = "Moderate Risk";
                indicatorClass = "result-indicator moderate";
                indicatorSymbol = "!";
                message = "Based on the provided parameters, there is a moderate risk of diabetes. Consider discussing these results with a healthcare professional.";
            } else {
                riskLevel = "High Risk";
                indicatorClass = "result-indicator positive";
                indicatorSymbol = "!";
                message = "Based on the provided parameters, there is a significant risk of diabetes. Please consult with a healthcare professional for proper medical advice.";
            }

            // Update the UI elements
            resultIndicator.className = indicatorClass;
            resultIndicator.innerHTML = indicatorSymbol;
            resultText.textContent = `Prediction Result: ${riskLevel}`;
            probabilityText.textContent = `${probability}% probability`;
            messageText.textContent = message;
            
            // Show result container
            form.classList.add('hidden');
            resultContainer.classList.remove('hidden');
        
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Please try again later.');
            loader.classList.add('hidden');
        }
    });

    // Reset button handler
    resetBtn.addEventListener('click', () => {
        form.reset();
        form.classList.remove('hidden');
        resultContainer.classList.add('hidden');
    });
});