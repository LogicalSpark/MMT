package eu.modernmt.model.impl.tmx;

import eu.modernmt.model.BilingualCorpus;

import javax.xml.stream.Location;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by davide on 14/03/16.
 */
class TMXPairReader {

    private final SimpleDateFormat dateFormat = new SimpleDateFormat("YYYYMMdd'T'HHmmss'Z'");

    private BilingualCorpus.StringPair wrap(XMLStreamReader reader, String source, String target, Date timestamp) throws XMLStreamException {
        if (source == null || (source = source.trim()).isEmpty()) {
            Location location = reader.getLocation();
            throw new XMLStreamException("Missing source sentence near line " + location.getLineNumber());
        }

        if (target == null || (target = target.trim()).isEmpty()) {
            Location location = reader.getLocation();
            throw new XMLStreamException("Missing target sentence near line " + location.getLineNumber());
        }

        return new BilingualCorpus.StringPair(source.replace('\n', ' '), target.replace('\n', ' '), timestamp);
    }

    public BilingualCorpus.StringPair read(XMLStreamReader reader, String sourceLanguage, String targetLanguage) throws XMLStreamException {
        while (reader.hasNext()) {
            int type = reader.next();

            switch (type) {
                case XMLStreamReader.START_ELEMENT:
                    if ("tu".equals(reader.getLocalName()))
                        return readTu(reader, sourceLanguage, targetLanguage);
                    break;
            }
        }

        return null;
    }

    private BilingualCorpus.StringPair readTu(XMLStreamReader reader, String sourceLanguage, String targetLanguage) throws XMLStreamException {
        Date timestamp = null;

        String date = reader.getAttributeValue(null, "changedate");
        if (date == null)
            date = reader.getAttributeValue(null, "creationdate");

        if (date != null) {
            try {
                timestamp = dateFormat.parse(date);
            } catch (ParseException | NumberFormatException e) {
                throw new XMLStreamException("Invalid date '" + date + "' at line " + reader.getLocation().getLineNumber(), e);
            }
        }

        String source = null;
        String target = null;

        while (reader.hasNext()) {
            int type = reader.next();

            switch (type) {
                case XMLStreamReader.START_ELEMENT:
                    if ("tuv".equals(reader.getLocalName())) {
                        String lang = reader.getAttributeValue(null, "lang");
                        String text = readTuv(reader);

                        if (lang == null) {
                            throw new XMLStreamException("Missing language for 'tuv' at line " + reader.getLocation().getLineNumber());
                        } else if (lang.startsWith(sourceLanguage)) {
                            source = text;
                        } else if (lang.startsWith(targetLanguage)) {
                            target = text;
                        } else {
                            throw new XMLStreamException("Invalid language code found: " + lang);
                        }
                    }
                    break;
                case XMLStreamReader.END_ELEMENT:
                    if ("tu".equals(reader.getLocalName())) {
                        return wrap(reader, source, target, timestamp);
                    }
                    break;
            }
        }

        throw new XMLStreamException("Missing closing tag for 'tuv' element at line " + reader.getLocation().getLineNumber());
    }

    private String readTuv(XMLStreamReader reader) throws XMLStreamException {
        while (reader.hasNext()) {
            int type = reader.next();

            switch (type) {
                case XMLStreamReader.START_ELEMENT:
                    if ("seg".equals(reader.getLocalName())) {
                        return readCharacters(reader);
                    }
                    break;
            }
        }

        throw new XMLStreamException("Missing 'seg' inside 'tuv' element at line " + reader.getLocation().getLineNumber());
    }

    private String readCharacters(XMLStreamReader reader) throws XMLStreamException {
        StringBuilder result = new StringBuilder();

        while (reader.hasNext()) {
            int type = reader.next();

            switch (type) {
                case XMLStreamReader.CHARACTERS:
                case XMLStreamReader.CDATA:
                    result.append(reader.getText());
                    break;
                case XMLStreamReader.END_ELEMENT:
                    return result.toString();
            }
        }

        throw new XMLStreamException("Premature end of file");
    }


}
