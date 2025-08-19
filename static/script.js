/*
 *
 * ================================================================================
 * Author: Andrea Iommi
 * Code Ownership:
 *     - All Python source code in this file is written solely by the author.
 * Documentation Notice:
 *     - All docstrings and inline documentation are written by ChatGPT,
 *       but thoroughly checked and approved by the author for accuracy.
 * ================================================================================
 */

/**
 * Fetches the current training progress from the server and updates the UI.
 *
 * This function calls the Flask endpoint `/progress` every second until training
 * is complete. It updates the progress bar, status text, and training metrics
 * displayed on the web page.
 *
 * Updates the following HTML elements:
 * - Progress bar width (#bar)
 * - Training status percentage (#status)
 * - Training loss mean (#tr_loss_mean)
 * - Training loss std (#tr_loss_std)
 * - Validation loss mean (#vl_loss_mean)
 * - Validation loss std (#vl_loss_std)
 * - Validation F1 score (#vl_f1)
 * - Validation accuracy (#vl_acc)
 * - Training start time (#time_start)
 * - Training elapsed time (#time_elapsed)
 * - Current epoch (#epoch)
 * - Current batch (#batch)
 *
 * Once the training status reaches 100%, the Start Training button is re-enabled,
 * its text is reset, and a completion alert is shown.
 */
function updateProgress() {
    // Fetch progress JSON from the server
    fetch('/progress')
        .then(res => res.json())
        .then(data => {
            // Update progress bar width and percentage text
            document.getElementById("bar").style.width = data.status + "%";
            document.getElementById("status").innerText = data.status + "%";

            // Update training metrics
            document.getElementById("tr_loss_mean").innerText = data.tr_loss_mean;
            document.getElementById("tr_loss_std").innerText = data.tr_loss_std;

            // Update validation metrics
            document.getElementById("vl_loss_mean").innerText = data.vl_loss_mean;
            document.getElementById("vl_loss_std").innerText = data.vl_loss_std;
            document.getElementById("vl_f1").innerText = data.vl_f1;
            document.getElementById("vl_acc").innerText = data.vl_acc;

            // Update training timing info
            document.getElementById("time_start").innerText = data.time_start;
            document.getElementById("time_elapsed").innerText = data.time_elapsed;

            // Update epoch and batch info
            document.getElementById("epoch").innerText = data.epoch;
            document.getElementById("batch").innerText = data.batch;

            // Continue polling every second if training is not complete
            if (data.status < 100) {
                setTimeout(updateProgress, 1000);
            } else {
                // Training completed: re-enable Start Training button and alert the user
                document.getElementById("trainBtn").disabled = false;
                document.getElementById("trainBtn").innerText = "Start Training";
                alert("Training completato!");
                // Here you can also update charts or other visualizations if needed
            }
        });
}
