package eu.modernmt.facade;

import eu.modernmt.aligner.Aligner;
import eu.modernmt.aligner.AlignerException;
import eu.modernmt.cluster.ClusterNode;
import eu.modernmt.cluster.TranslationTask;
import eu.modernmt.cluster.error.SystemShutdownException;
import eu.modernmt.context.ContextAnalyzer;
import eu.modernmt.context.ContextAnalyzerException;
import eu.modernmt.decoder.Decoder;
import eu.modernmt.decoder.DecoderException;
import eu.modernmt.decoder.DecoderWithNBest;
import eu.modernmt.engine.Engine;
import eu.modernmt.lang.Language;
import eu.modernmt.lang.LanguageIndex;
import eu.modernmt.lang.LanguagePair;
import eu.modernmt.lang.UnsupportedLanguageException;
import eu.modernmt.model.Alignment;
import eu.modernmt.model.ContextVector;
import eu.modernmt.model.Sentence;
import eu.modernmt.model.Translation;
import eu.modernmt.model.corpus.Corpus;
import eu.modernmt.model.corpus.impl.StringCorpus;
import eu.modernmt.model.corpus.impl.parallel.FileCorpus;
import eu.modernmt.processing.Postprocessor;
import eu.modernmt.processing.Preprocessor;
import eu.modernmt.processing.ProcessingException;
import eu.modernmt.processing.splitter.SentenceSplitter;
import eu.modernmt.processing.splitter.TranslationJoiner;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * Created by davide on 31/01/17.
 */
public class TranslationFacade {

    private static final Logger logger = LogManager.getLogger(TranslationFacade.class);
    private LanguagePair lastTranslationLanguage = null;

    public enum Priority {
        HIGH(0), NORMAL(1), BACKGROUND(2);  //three priority values are allowed

        public final int intValue;

        Priority(int value) {
            this.intValue = value;
        }
    }

    // =============================
    //  Translation
    // =============================

    public Translation get(UUID user, LanguagePair direction, String sentence, Priority priority) throws ProcessingException, DecoderException, AlignerException {
        return get(user, direction, sentence, null, 0, priority);
    }

    public Translation get(UUID user, LanguagePair direction, String sentence, ContextVector translationContext, Priority priority) throws ProcessingException, DecoderException, AlignerException {
        return get(user, direction, sentence, translationContext, 0, priority);
    }

    public Translation get(UUID user, LanguagePair direction, String sentence, int nbest, Priority priority) throws ProcessingException, DecoderException, AlignerException {
        return get(user, direction, sentence, null, nbest, priority);
    }

    public Translation get(UUID user, LanguagePair direction, String sentence, ContextVector translationContext, int nbest, Priority priority) throws ProcessingException, DecoderException, AlignerException {
        direction = mapLanguagePair(direction);
        if (nbest > 0)
            ensureDecoderSupportsNBest();

        TranslationTask task = new TranslationTaskImpl(user, direction, sentence, translationContext, nbest, priority);

        try {
            ClusterNode node = ModernMT.getNode();

            Future<Translation> future = node.submit(task);
            Translation translation = future.get();

            if (logger.isDebugEnabled())
                logger.debug("Translation of " + translation.getSource().length() + " words took " + (((double) translation.getElapsedTime()) / 1000.) + "s");

            return translation;
        } catch (InterruptedException e) {
            throw new SystemShutdownException(e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();

            if (cause instanceof ProcessingException)
                throw (ProcessingException) cause;
            else if (cause instanceof DecoderException)
                throw (DecoderException) cause;
            else if (cause instanceof AlignerException)
                throw (AlignerException) cause;
            else if (cause instanceof RuntimeException)
                throw (RuntimeException) cause;
            else
                throw new Error("Unexpected exception thrown: " + cause.getMessage(), cause);
        }
    }

    public void test() throws DecoderException, AlignerException, ProcessingException {
        LanguagePair language = selectForTest();
        String text = "Translation test " + new Random().nextInt();

        TranslationTaskImpl task = new TranslationTaskImpl(null, language, text, null, 0, TranslationFacade.Priority.HIGH);
        Translation translation = task.call();
        if (!translation.hasWords())
            throw new DecoderException("Empty translation for test sentence '" + text + "'");
    }

    private LanguagePair selectForTest() {
        LanguagePair language = getLastTranslationLanguage();

        if (language == null) {
            LanguageIndex index = ModernMT.getNode().getEngine().getLanguageIndex();

            for (LanguagePair pair : index.getLanguages()) {
                if ("en".equalsIgnoreCase(pair.source.getLanguage()))
                    return pair;
            }

            language = index.getLanguages().iterator().next();
        }

        return language;
    }

    private synchronized LanguagePair getLastTranslationLanguage() {
        return this.lastTranslationLanguage;
    }

    private synchronized void setLastTranslationLanguage(LanguagePair lastTranslationLanguage) {
        this.lastTranslationLanguage = lastTranslationLanguage;
    }

    // =============================
    //  Context Vector
    // =============================

    public ContextVector getContextVector(UUID user, LanguagePair direction, File context, int limit) throws ContextAnalyzerException {
        direction = mapLanguagePair(direction);

        Engine engine = ModernMT.getNode().getEngine();
        ContextAnalyzer analyzer = engine.getContextAnalyzer();

        return analyzer.getContextVector(user, direction, context, limit);
    }

    public ContextVector getContextVector(UUID user, LanguagePair direction, String context, int limit) throws ContextAnalyzerException {
        direction = mapLanguagePair(direction);

        Engine engine = ModernMT.getNode().getEngine();
        ContextAnalyzer analyzer = engine.getContextAnalyzer();

        return analyzer.getContextVector(user, direction, context, limit);
    }

    public Map<Language, ContextVector> getContextVectors(UUID user, File context, int limit, Language source, Language... targets) throws ContextAnalyzerException {
        return getContextVectors(user, new FileCorpus(context, null, source), limit, source, targets);
    }

    public Map<Language, ContextVector> getContextVectors(UUID user, String context, int limit, Language source, Language... targets) throws ContextAnalyzerException {
        return getContextVectors(user, new StringCorpus(null, source, context), limit, source, targets);
    }

    private Map<Language, ContextVector> getContextVectors(UUID user, Corpus context, int limit, Language source, Language... targets) throws ContextAnalyzerException {
        Engine engine = ModernMT.getNode().getEngine();
        ContextAnalyzer analyzer = engine.getContextAnalyzer();

        HashMap<Language, ContextVector> result = new HashMap<>(targets.length);
        for (Language target : targets) {
            try {
                LanguagePair direction = mapLanguagePair(new LanguagePair(source, target));
                ContextVector contextVector = analyzer.getContextVector(user, direction, context, limit);
                result.put(target, contextVector);
            } catch (UnsupportedLanguageException e) {
                // ignore it
            }
        }

        return result;
    }

    // -----------------------------
    //  Util functions
    // -----------------------------

    private void ensureDecoderSupportsNBest() {
        Decoder decoder = ModernMT.getNode().getEngine().getDecoder();
        if (!(decoder instanceof DecoderWithNBest))
            throw new UnsupportedOperationException("Decoder '" + decoder.getClass().getSimpleName() + "' does not support N-best.");
    }

    private LanguagePair mapLanguagePair(LanguagePair pair) {
        LanguageIndex index = ModernMT.getNode().getEngine().getLanguageIndex();

        LanguagePair mapped = index.map(pair, true);
        if (mapped == null)
            throw new UnsupportedLanguageException(pair);

        return mapped;
    }

    // -----------------------------
    //  Internal Operations
    // -----------------------------

    private static class TranslationTaskImpl implements TranslationTask {

        public final UUID user;
        public final LanguagePair direction;
        public final String text;
        public final ContextVector context;
        public final int nbest;
        public final Priority priority;

        public TranslationTaskImpl(UUID user, LanguagePair direction, String text, ContextVector context, int nbest, Priority priority) {
            this.user = user;
            this.direction = direction;
            this.text = text;
            this.context = context;
            this.nbest = nbest;
            this.priority = priority;
        }

        @Override
        public Translation call() throws ProcessingException, DecoderException, AlignerException {
            ModernMT.translation.setLastTranslationLanguage(direction);

            ClusterNode node = ModernMT.getNode();

            Engine engine = node.getEngine();
            Decoder decoder = engine.getDecoder();
            Preprocessor preprocessor = engine.getPreprocessor();
            Postprocessor postprocessor = engine.getPostprocessor();

            long begin = System.currentTimeMillis();

            Sentence sentence = preprocessor.process(direction, text);
            Translation translation;

            // Sentence splitter
            Sentence[] sentencePieces = SentenceSplitter.forLanguage(direction.source).split(sentence);
            Translation[] translationPieces = translate(sentencePieces, decoder);

            translation = this.merge(sentence, sentencePieces, translationPieces);

            postprocessor.process(direction, translation);

            // NBest list
            if (translation.hasNbest()) {
                List<Translation> hypotheses = translation.getNbest();
                postprocessor.process(direction, hypotheses);
            }

            translation.setElapsedTime(System.currentTimeMillis() - begin);

            return translation;
        }

        private Translation[] translate(Sentence[] sentences, Decoder decoder)
                throws DecoderException, AlignerException {
            Translation[] translations = new Translation[sentences.length];

            for (int i = 0; i < sentences.length; i++)
                translations[i] = this.translate(sentences[i], decoder);

            return translations;
        }

        private Translation translate(Sentence sentence, Decoder decoder) throws DecoderException, AlignerException {
            Translation translation;

            if (nbest > 0) {
                DecoderWithNBest nBestDecoder = (DecoderWithNBest) decoder;
                translation = nBestDecoder.translate(user, direction, sentence, context, nbest);
            } else {
                translation = decoder.translate(user, direction, sentence, context);
            }

            // Compute alignments if missing

            if (!translation.hasAlignment()) {
                Engine engine = ModernMT.getNode().getEngine();
                Aligner aligner = engine.getAligner();

                Alignment alignment = aligner.getAlignment(direction, sentence, translation);
                translation.setWordAlignment(alignment);

                if (translation.hasNbest()) {
                    for (Translation nbest : translation.getNbest()) {
                        if (!nbest.hasAlignment()) {
                            Alignment nbestAlignment = aligner.getAlignment(direction, sentence, nbest);
                            nbest.setWordAlignment(nbestAlignment);
                        }
                    }
                }
            }

            return translation;
        }

        private Translation merge(Sentence originalSentence, Sentence[] sentencePieces, Translation[] translationPieces) {
            Translation translation = TranslationJoiner.join(originalSentence, sentencePieces, translationPieces);

            if (translation.hasNbest()) {
                int nbestSize = 0;
                for (Translation piece : translationPieces)
                    nbestSize = Math.max(nbestSize, piece.getNbest().size());

                List<Translation> globalNBests = new ArrayList<>(nbestSize);
                Translation[] ithNBests = new Translation[translationPieces.length];

                for (int i = 0; i < nbestSize; i++) {
                    for (int t = 0; t < translationPieces.length; t++) {
                        Translation piece = translationPieces[t];
                        int index = Math.min(i, piece.length() - 1);  // If not enough options, take the last one

                        ithNBests[t] = piece.getNbest().get(index);
                    }

                    globalNBests.add(TranslationJoiner.join(originalSentence, sentencePieces, ithNBests));
                }

                translation.setNbest(globalNBests);
            }

            return translation;
        }

        @Override
        public int getPriority() {
            return this.priority.intValue;
        }

        @Override
        public LanguagePair getLanguage() {
            return direction;
        }
    }
}
