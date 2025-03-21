package com.graphsum.visualization;

public class LoadingSpinner implements Runnable {
    private volatile boolean running = true;

    public void stop() {
        running = false;
    }

    @Override
    public void run() {
        String[] spinner = {"|", "/", "-", "\\"};
        int i = 0;
        while (running) {
            System.out.print("\rProcessing " + spinner[i % spinner.length]);
            i++;
            try {
                Thread.sleep(200); // Ajusta la velocidad del spinner
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}