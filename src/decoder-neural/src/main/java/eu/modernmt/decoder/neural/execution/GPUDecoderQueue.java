package eu.modernmt.decoder.neural.execution;

import eu.modernmt.decoder.neural.natv.NativeProcess;

import java.io.IOException;

class GPUDecoderQueue extends DecoderQueueImpl {

    private final int[] gpus;
    private int idx;

    GPUDecoderQueue(NativeProcess.Builder processBuilder, int[] gpus) {
        super(processBuilder, gpus.length);

        this.gpus = gpus;
        this.idx = this.gpus.length - 1;
    }

    @Override
    protected final NativeProcess startProcess(NativeProcess.Builder builder) throws IOException {
        int gpu;

        synchronized (this) {
            gpu = this.gpus[this.idx--];
        }

        return builder.startOnGPU(gpu);
    }

    @Override
    protected final synchronized void onProcessDied(NativeProcess process) {
        synchronized (this) {
            this.gpus[++this.idx] = process.getGPU();
        }
    }

}
