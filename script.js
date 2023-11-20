// script.js
$(document).ready(function() {
    $('form').submit(function(event) {
        event.preventDefault();

        var selectedDate = $('#selected-date').val();

        $.ajax({
            url: 'predict.php',
            method: 'POST',
            data: { 'selected-date': selectedDate },
            dataType: 'json',
            success: function(response) {
                displayBarGraph(response);
            }
        });
    });

    function displayBarGraph(predictedWastage) {
        var labels = Object.keys(predictedWastage);
        var data = Object.values(predictedWastage);

        var ctx = document.getElementById('chart').getContext('2d');
        var chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Food Wastage',
                    data: data,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        stepSize: 1
                    }
                }
            }
        });
    }
});
