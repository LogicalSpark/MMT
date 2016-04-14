package eu.modernmt.aligner.fastalign;

import eu.modernmt.aligner.Aligner;
import eu.modernmt.aligner.symal.Symmetrisation;
import eu.modernmt.model.Sentence;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;

/**
 * Created by lucamastrostefano on 14/03/16.
 */
public class SymmetrizedAligner implements Aligner {

    private static final String forwardModelFileName = "model.align.fwd";
    private static final String backwardModelFileName = "model.align.bwd";
    private static final Logger logger = LogManager.getLogger(eu.modernmt.aligner.fastalign.SymmetrizedAligner.class);
    public static final Symmetrisation.Strategy DEFAULT_SYMMETRIZATION_STRATEGY = Symmetrisation.Strategy.GrowDiagFinalAnd;

    private static class AlignerInitializer implements Runnable {

        private Aligner aligner;

        public AlignerInitializer(Aligner aligner) {
            this.aligner = aligner;
        }

        @Override
        public void run() {
            try {
                this.aligner.init();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    private final FastAlign forwardAlignerProcess;
    private final FastAlign backwardAlignerProcess;
    private Symmetrisation.Strategy simmetrizationStrategy;

    public SymmetrizedAligner(String enginePath) {
        File modelsFile = new File(enginePath,
                "models" + File.separatorChar + "phrase_tables");
        String forwardModelFilePath = new File(modelsFile.getAbsolutePath(), forwardModelFileName).getAbsolutePath();
        String backwardModelFilePath = new File(modelsFile.getAbsolutePath(), backwardModelFileName).getAbsolutePath();

        forwardAlignerProcess = new FastAlign(false, forwardModelFilePath);
        backwardAlignerProcess = new FastAlign(true, backwardModelFilePath);
        this.simmetrizationStrategy = DEFAULT_SYMMETRIZATION_STRATEGY;
    }

    @Override
    public void init() {
        try {
            logger.info("Initializing Fast Align");
            Thread forwardModelInitializer = new Thread(new AlignerInitializer(this.forwardAlignerProcess));
            Thread backwardModelInitializer = new Thread(new AlignerInitializer(this.backwardAlignerProcess));
            logger.info("Loading forward model");
            forwardModelInitializer.start();
            logger.info("Loading backward model");
            backwardModelInitializer.start();
            forwardModelInitializer.join();
            backwardModelInitializer.join();
            logger.info("Fast Align initialized");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void setSymmetrizationStrategy(Symmetrisation.Strategy strategy) {
        this.simmetrizationStrategy = strategy;
    }

    @Override
    public synchronized int[][] getAlignments(Sentence sentence, Sentence translation) throws IOException {
        int numberOfSentenceWords = sentence.getWords().length;
        int numberOfTranslationWords = translation.getWords().length;
        int[][] forwardAlignments = this.forwardAlignerProcess.getAlignments(sentence, translation);
        forwardAlignments = FastAlign.interpolateAlignments(forwardAlignments, numberOfSentenceWords,
                numberOfTranslationWords);
        int[][] backwardAlignments = this.backwardAlignerProcess.getAlignments(sentence, translation);
        backwardAlignments = FastAlign.interpolateAlignments(backwardAlignments, numberOfSentenceWords,
                numberOfTranslationWords);

        int[][] invertedBackwardAlignments = eu.modernmt.aligner.fastalign.SymmetrizedAligner.invertAlignments(backwardAlignments);
        logger.debug("Symmetrising");
        String symmetrisedAlignments = Symmetrisation.symmetriseMosesFormatAlignment(forwardAlignments,
                invertedBackwardAlignments, this.simmetrizationStrategy);
        logger.debug("Symmetrised alignments: " + symmetrisedAlignments);
        return FastAlign.parseAlignments(symmetrisedAlignments);
    }

    private static int[][] invertAlignments(int[][] alignments) {
        int[][] invertedAlignments = new int[alignments.length][2];
        for (int i = 0; i < alignments.length; i++) {
            int[] alignment = alignments[i];
            invertedAlignments[i] = new int[]{alignment[1], alignment[0]};
        }
        Arrays.sort(invertedAlignments, new Comparator<int[]>() {
            @Override
            public int compare(int[] a1, int[] a2) {
                int c = a1[0] - a2[0];
                if (c == 0) {
                    c = a1[1] - a2[1];
                }
                return c;
            }
        });
        return invertedAlignments;
    }

    @Override
    public void close() throws IOException {
        this.forwardAlignerProcess.close();
        this.backwardAlignerProcess.close();
    }

}
