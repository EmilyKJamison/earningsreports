package de.tudarmstadt.ukp.experiments.ej.ieearningsreports.FeatureExtractors;

import java.util.Set;

import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.tc.api.exception.TextClassificationException;
import de.tudarmstadt.ukp.dkpro.tc.api.features.DocumentFeatureExtractor;
import de.tudarmstadt.ukp.dkpro.tc.api.features.Feature;
import de.tudarmstadt.ukp.dkpro.tc.api.features.FeatureExtractorResource_ImplBase;

public class DocumentTextPrinterFE
	extends FeatureExtractorResource_ImplBase
	implements DocumentFeatureExtractor
{

	@Override
	public Set<Feature> extract(JCas jcas)
		throws TextClassificationException
	{
//		System.out.println("Text: " + jcas.getDocumentText());
		System.out.println("NEWDOCUMENT");
		for (Sentence s: JCasUtil.select(jcas, Sentence.class)){
			System.out.println(s.getCoveredText().replace("\n", "") + "\t" + "PUBINFO HEADLINE HIGHLIGHTS "
					+ "LONGHIGHLIGHTS PARATEXT TABLE SECTIONHEADER ABOUTCOMPANY LEGALESE FOOTNOTE QUOTE SPEAKER\n");
		}
		
		return new Feature("SampleFeature", 0).asSet();
	}

}
