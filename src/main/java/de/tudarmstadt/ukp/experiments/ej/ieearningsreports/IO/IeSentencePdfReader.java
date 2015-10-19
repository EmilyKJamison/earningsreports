/**
 * Copyright 2015
 * Ubiquitous Knowledge Processing (UKP) Lab
 * Technische Universität Darmstadt
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */
package de.tudarmstadt.ukp.experiments.ej.ieearningsreports.IO;

import static org.apache.commons.io.IOUtils.closeQuietly;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.uima.collection.CollectionException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.factory.JCasBuilder;
import org.apache.uima.jcas.JCas;

import de.tudarmstadt.ukp.dkpro.core.api.io.JCasResourceCollectionReader_ImplBase;
import de.tudarmstadt.ukp.dkpro.core.api.parameter.ComponentParameters;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Document;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.tc.api.io.TCReaderSequence;
import de.tudarmstadt.ukp.dkpro.tc.api.type.TextClassificationOutcome;
import de.tudarmstadt.ukp.dkpro.tc.api.type.TextClassificationSequence;
import de.tudarmstadt.ukp.dkpro.tc.api.type.TextClassificationUnit;

/**
 * Reads our Earnings Reports as sequential sentences in a document.
 *
 */
public class IeSentencePdfReader
    extends JCasResourceCollectionReader_ImplBase
	implements TCReaderSequence

{
    private static final int SENTENCE = 0;
    private static final int LABELARRAY = 1;


    /**
     * Character encoding of the input data.
     */
    public static final String PARAM_ENCODING = ComponentParameters.PARAM_SOURCE_ENCODING;
    @ConfigurationParameter(name = PARAM_ENCODING, mandatory = true, defaultValue = "UTF-8")
    private String encoding;

    /**
     * The language.
     */
    public static final String PARAM_LANGUAGE = ComponentParameters.PARAM_LANGUAGE;
    @ConfigurationParameter(name = PARAM_LANGUAGE, mandatory = true)
    private String language;
    
    /**
     * Load the chunk tag to UIMA type mapping from this location instead of locating
     * the mapping automatically.
     */
    public static final String PARAM_CHUNK_MAPPING_LOCATION = ComponentParameters.PARAM_CHUNK_MAPPING_LOCATION;
    @ConfigurationParameter(name = PARAM_CHUNK_MAPPING_LOCATION, mandatory = false)
    protected String chunkMappingLocation;
    
    
    @Override
    public void getNext(JCas aJCas)
        throws IOException, CollectionException
    {
        
        Resource res = nextFile();
        initCas(aJCas, res);
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(res.getInputStream(), encoding));
            convert(aJCas, reader);
        }
        finally {
            closeQuietly(reader);
        }
    }

    private void convert(JCas aJCas, BufferedReader aReader)
        throws IOException
    {
        JCasBuilder doc = new JCasBuilder(aJCas);
        
        List<String[]> sentencesAndLabels;
        while ((sentencesAndLabels = readOneDocument(aReader)) != null) {
            if (sentencesAndLabels.isEmpty()) {
                break; 
            }

            int documentBegin = doc.getPosition();
            int documentEnd = documentBegin;

            List<Sentence> sentences = new ArrayList<Sentence>();
            
            
            for (String[] sentenceAndLabel : sentencesAndLabels) {
                Sentence sentence = doc.add(sentenceAndLabel[SENTENCE], Sentence.class);
                documentEnd = sentence.getEnd();
                doc.add(" ");
                
                TextClassificationUnit unit = new TextClassificationUnit(aJCas, sentence.getBegin(), sentence.getEnd());
                unit.addToIndexes();
                
                for (String label: sentenceAndLabel[LABELARRAY].split(" ")){
                	TextClassificationOutcome outcome = new TextClassificationOutcome(aJCas, sentence.getBegin(), sentence.getEnd());
                	outcome.setOutcome(label);
                	outcome.addToIndexes();
                	break; //FIXME!!! Turns out TC doesn't support multilabel sequential learning, so we just use the first label for now.
                }
            
                sentences.add(sentence);
            }

            // Add sent to doc
            Document document = new Document(aJCas, documentBegin, documentEnd);
            document.addToIndexes();
            
            TextClassificationSequence sequence = new TextClassificationSequence(aJCas, documentBegin, documentEnd);
            sequence.addToIndexes();

            // Once sentence per line. 
            doc.add("\n");
        }

        doc.close();
    }

    /**
     * Read a single document.
     */
    private static List<String[]> readOneDocument(BufferedReader aReader)
        throws IOException
    {
        List<String[]> sentencesAndLabels = new ArrayList<String[]>();
        String line;
        while ((line = aReader.readLine()) != null) {
            if (line.equals("ENDDOCUMENT")) {
                break; // End of document
            }
            if (line.length() == 0) {
            	continue;
            }
            
            String[] fields = line.split("\t");
            if (fields.length != 2) {
            	System.out.println(line);
                throw new IOException(
                        "Invalid file format. Line needs to have 2 tab-separted fields: "
                        + "a sentence and a list of space-separated labels");
            }
            sentencesAndLabels.add(fields);
        }
        return sentencesAndLabels;
    }

	@Override
	public String getTextClassificationOutcome(JCas jcas,
			TextClassificationUnit unit) throws CollectionException
	{
		// without function here, as we do not represent this in the CAS
		return null;
	}
}