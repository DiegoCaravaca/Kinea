function update() {
    fetch('/data')
        .then(res => res.json())
        .then(data => {
            document.getElementById('state').innerText = data.state;
            document.getElementById('pred').innerText = data.prediction;
            document.getElementById('rms-text').innerText = data.rms + " V";

            // Bar width as a share of VREF (x2 so small signals stay visible)
            let percent = (data.rms / 3.3) * 100 * 2;
            document.getElementById('rms-bar').style.width = Math.min(percent, 100) + "%";
        });
}

// Refresh every 100 ms
setInterval(update, 100);
