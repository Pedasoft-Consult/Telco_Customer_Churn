// JavaScript functionality for Customer Churn Prediction application

document.addEventListener('DOMContentLoaded', function() {
    // Handle Internet Service selection
    const internetServiceSelect = document.getElementById('InternetService');
    if (internetServiceSelect) {
        internetServiceSelect.addEventListener('change', function() {
            const internetRelatedSelects = [
                'OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies'
            ];

            // If "No" internet service is selected, set all internet-related services to "No internet service"
            if (this.value === 'No') {
                internetRelatedSelects.forEach(selectId => {
                    const select = document.getElementById(selectId);
                    if (select) {
                        // Find "No internet service" option
                        const noInternetOption = Array.from(select.options).find(
                            option => option.value === 'No internet service'
                        );

                        if (noInternetOption) {
                            select.value = 'No internet service';
                            select.disabled = true;
                        }
                    }
                });
            } else {
                // Re-enable internet-related selects
                internetRelatedSelects.forEach(selectId => {
                    const select = document.getElementById(selectId);
                    if (select) {
                        select.disabled = false;
                        // Reset to default if currently set to "No internet service"
                        if (select.value === 'No internet service') {
                            select.value = '';
                        }
                    }
                });
            }
        });
    }

    // Handle Phone Service selection
    const phoneServiceSelect = document.getElementById('PhoneService');
    if (phoneServiceSelect) {
        phoneServiceSelect.addEventListener('change', function() {
            const multipleLinesSelect = document.getElementById('MultipleLines');

            if (multipleLinesSelect) {
                if (this.value === 'No') {
                    // If no phone service, set multiple lines to "No phone service" and disable
                    const noPhoneOption = Array.from(multipleLinesSelect.options).find(
                        option => option.value === 'No phone service'
                    );

                    if (noPhoneOption) {
                        multipleLinesSelect.value = 'No phone service';
                        multipleLinesSelect.disabled = true;
                    }
                } else {
                    // Enable multiple lines selection
                    multipleLinesSelect.disabled = false;

                    // Reset if currently set to "No phone service"
                    if (multipleLinesSelect.value === 'No phone service') {
                        multipleLinesSelect.value = '';
                    }
                }
            }
        });
    }

    // Calculate monthly charges automatically based on services
    const calculateCharges = function() {
        // Base price components
        const basePrices = {
            'PhoneService': 20,
            'MultipleLines': 10,
            'InternetService': {
                'DSL': 25,
                'Fiber optic': 50,
                'No': 0
            },
            'OnlineSecurity': 10,
            'OnlineBackup': 10,
            'DeviceProtection': 10,
            'TechSupport': 15,
            'StreamingTV': 15,
            'StreamingMovies': 15
        };

        // Contract discounts
        const contractDiscount = {
            'Month-to-month': 0,
            'One year': 0.1, // 10% discount
            'Two year': 0.2  // 20% discount
        };

        // Start with base charge
        let calculatedCharge = 0;

        // Add phone service if selected
        const phoneService = document.getElementById('PhoneService');
        if (phoneService && phoneService.value === 'Yes') {
            calculatedCharge += basePrices.PhoneService;

            // Add multiple lines if selected
            const multipleLines = document.getElementById('MultipleLines');
            if (multipleLines && multipleLines.value === 'Yes') {
                calculatedCharge += basePrices.MultipleLines;
            }
        }

        // Add internet service
        const internetService = document.getElementById('InternetService');
        if (internetService && internetService.value !== '') {
            calculatedCharge += basePrices.InternetService[internetService.value];

            // Add internet-related services if internet is selected
            if (internetService.value !== 'No') {
                const internetServices = [
                    'OnlineSecurity',
                    'OnlineBackup',
                    'DeviceProtection',
                    'TechSupport',
                    'StreamingTV',
                    'StreamingMovies'
                ];

                internetServices.forEach(service => {
                    const serviceSelect = document.getElementById(service);
                    if (serviceSelect && serviceSelect.value === 'Yes') {
                        calculatedCharge += basePrices[service];
                    }
                });
            }
        }

        // Apply contract discount
        const contract = document.getElementById('Contract');
        if (contract && contract.value !== '') {
            calculatedCharge = calculatedCharge * (1 - contractDiscount[contract.value]);
        }

        // Update monthly charges field
        const monthlyCharges = document.getElementById('MonthlyCharges');
        if (monthlyCharges) {
            monthlyCharges.value = calculatedCharge.toFixed(2);
        }
    };

    // Auto-calculate charges when service options change
    const serviceInputs = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract'
    ];

    serviceInputs.forEach(inputId => {
        const input = document.getElementById(inputId);
        if (input) {
            input.addEventListener('change', calculateCharges);
        }
    });

    // Form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(event) {
            let isValid = true;
            const requiredSelects = document.querySelectorAll('select[required]');
            const requiredInputs = document.querySelectorAll('input[required]');

            // Check all required selects
            requiredSelects.forEach(select => {
                if (select.value === '') {
                    isValid = false;
                    select.classList.add('is-invalid');
                } else {
                    select.classList.remove('is-invalid');
                }
            });

            // Check all required inputs
            requiredInputs.forEach(input => {
                if (input.value === '') {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });

            if (!isValid) {
                event.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    }
});