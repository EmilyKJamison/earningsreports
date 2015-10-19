package de.tudarmstadt.ukp.experiments.ej.ieearningsreports.IO;

import java.io.IOException;

import org.apache.uima.cas.CAS;
import org.apache.uima.cas.CASException;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.jcas.JCas;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.io.pdf.PdfReader;
import de.tudarmstadt.ukp.dkpro.tc.api.io.TCReaderSingleLabel;
import de.tudarmstadt.ukp.dkpro.tc.api.type.TextClassificationOutcome;

public class IeOriginalPdfReader
	extends PdfReader
    implements TCReaderSingleLabel
{
    @Override
    public String getTextClassificationOutcome(JCas jcas)
            throws CollectionException
    {
            String uriString = DocumentMetaData.get(jcas).getDocumentUri();
            // Purposely nonsense outcomes; we just want to read the text.
            if (uriString.contains("WMT")){
            	return "positive";
            }else{
            	return "negative";
            }
    }
    @Override
    public void getNext(CAS aCAS)
        throws IOException, CollectionException
    {
        super.getNext(aCAS);
        
        JCas jcas;
        try {
            jcas = aCAS.getJCas();
        }
        catch (CASException e) {
            throw new CollectionException();
        }

        TextClassificationOutcome outcome = new TextClassificationOutcome(jcas);
        outcome.setOutcome(getTextClassificationOutcome(jcas));
        outcome.setWeight(getTextClassificationOutcomeWeight(jcas));
        outcome.addToIndexes();
    }


    /**
     * This methods adds a (default) weight to instances. Readers which assign specific weights to
     * instances need to override this method.
     * 
     * @param jcas
     *            the JCas to add the annotation to
     * @return a double between zero and one
     * @throws CollectionException
     */
	public double getTextClassificationOutcomeWeight(JCas jcas)
			throws CollectionException {
		return 1.0;
	}

}
