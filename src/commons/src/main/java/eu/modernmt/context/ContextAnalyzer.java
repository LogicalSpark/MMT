package eu.modernmt.context;

import eu.modernmt.data.DataBatch;
import eu.modernmt.data.DataListener;
import eu.modernmt.lang.LanguagePair;
import eu.modernmt.model.ContextVector;
import eu.modernmt.model.corpus.Corpus;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;

/**
 * Created by davide on 02/12/15.
 */
public interface ContextAnalyzer extends Closeable, DataListener {

    ContextVector getContextVector(UUID user, LanguagePair direction, String query, int limit) throws ContextAnalyzerException;

    ContextVector getContextVector(UUID user, LanguagePair direction, File source, int limit) throws ContextAnalyzerException;

    ContextVector getContextVector(UUID user, LanguagePair direction, Corpus query, int limit) throws ContextAnalyzerException;

    // Optimize the model for storage.
    // This operation can be extremely costly, use it only for backup purposes.
    void optimize() throws IOException;

    @Override
    void onDataReceived(DataBatch batch) throws ContextAnalyzerException;

    @Override
    Map<Short, Long> getLatestChannelPositions();

}
