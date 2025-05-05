function updateProgress() {
    fetch('/progress')
        .then(res => res.json())
        .then(data => {
            document.getElementById("bar").style.width = data.status + "%";
            document.getElementById("status").innerText = data.status + "%";

            document.getElementById("tr_loss_mean").innerText = data.tr_loss_mean
            document.getElementById("tr_loss_std").innerText = data.tr_loss_std
            document.getElementById("vl_loss_mean").innerText = data.vl_loss_mean
            document.getElementById("vl_loss_std").innerText = data.vl_loss_std
            document.getElementById("vl_f1").innerText = data.vl_f1
            document.getElementById("vl_acc").innerText = data.vl_acc
            document.getElementById("vl_f1").innerText = data.vl_f1
            document.getElementById("time_start").innerText = data.time_start
            document.getElementById("time_elapsed").innerText = data.time_elapsed
            document.getElementById("epoch").innerText = data.epoch
            document.getElementById("batch").innerText = data.batch

            if (data.status < 100) {
                setTimeout(updateProgress, 1000);
            } else {
                document.getElementById("trainBtn").disabled = false;
                document.getElementById("trainBtn").innerText = "Start Training";
                alert("Training completato!");
                // Qui puoi anche aggiornare i grafici, ecc.
            }
        });
}
