<!-- index.php -->
<html>
<head>
    <title>Food Wastage Prediction</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.5.1/chart.min.js"></script>
</head>
<body>
    <h1>Food Wastage Prediction</h1>
    <form method="POST" action="predict.php">
        <label for="selected-date">Select a date:</label>
        <input type="date" id="selected-date" name="selected-date" required>
        <button type="submit">Predict</button>
    </form>
    <div id="chart-container">
        <canvas id="chart"></canvas>
    </div>
    <script src="script.js"></script>
</body>
</html>
