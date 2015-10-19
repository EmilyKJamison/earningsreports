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
package de.tudarmstadt.ukp.experiments.ej.ieearningsreports;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.resource.ResourceInitializationException;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import de.tudarmstadt.ukp.dkpro.core.opennlp.OpenNlpPosTagger;
import de.tudarmstadt.ukp.dkpro.core.tokit.BreakIteratorSegmenter;
import de.tudarmstadt.ukp.dkpro.lab.Lab;
import de.tudarmstadt.ukp.dkpro.lab.task.BatchTask.ExecutionPolicy;
import de.tudarmstadt.ukp.dkpro.lab.task.Dimension;
import de.tudarmstadt.ukp.dkpro.lab.task.ParameterSpace;
import de.tudarmstadt.ukp.dkpro.tc.core.Constants;
import de.tudarmstadt.ukp.dkpro.tc.examples.util.DemoUtils;
import de.tudarmstadt.ukp.dkpro.tc.features.length.NrOfTokensDFE;
import de.tudarmstadt.ukp.dkpro.tc.features.length.NrOfTokensPerSentenceUFE;
import de.tudarmstadt.ukp.dkpro.tc.features.length.NrOfTokensUFE;
import de.tudarmstadt.ukp.dkpro.tc.features.ngram.LuceneNGramDFE;
import de.tudarmstadt.ukp.dkpro.tc.features.ngram.LuceneNGramUFE;
import de.tudarmstadt.ukp.dkpro.tc.features.ngram.base.NGramFeatureExtractorBase;
import de.tudarmstadt.ukp.dkpro.tc.crfsuite.CRFSuiteAdapter;
import de.tudarmstadt.ukp.dkpro.tc.crfsuite.CRFSuiteBatchCrossValidationReport;
import de.tudarmstadt.ukp.dkpro.tc.crfsuite.CRFSuiteClassificationReport;
import de.tudarmstadt.ukp.dkpro.tc.crfsuite.CRFSuiteOutcomeIDReport;
import de.tudarmstadt.ukp.dkpro.tc.crfsuite.writer.CRFSuiteDataWriter;
import de.tudarmstadt.ukp.dkpro.tc.ml.ExperimentCrossValidation;
import de.tudarmstadt.ukp.dkpro.tc.ml.report.BatchCrossValidationReport;
import de.tudarmstadt.ukp.dkpro.tc.ml.report.BatchStatisticsCVReport;
import de.tudarmstadt.ukp.dkpro.tc.weka.WekaClassificationAdapter;
import de.tudarmstadt.ukp.dkpro.tc.weka.WekaStatisticsClassificationAdapter;
import de.tudarmstadt.ukp.dkpro.tc.weka.report.WekaFeatureValuesReport;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.FeatureExtractors.LongWordsFeatureExtractor;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.FeatureExtractors.POSRatioFeatureExtractor;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.FeatureExtractors.PastVsFutureFeatureExtractor;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.FeatureExtractors.SuperlativeRatioFeatureExtractor;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.IO.IeSentencePdfReader;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.IO.ResultsMetricsPrinter;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.Utils.ExperimentUtils;

/**
 * This experiment classifies sentences extracted from Earnings Report Summaries, as: <br />
 * PUBINFO publication info, such as contact spokesperson <br />
 * HEADLINE of the document <br />
 * HIGHLIGHTS short bullet points about company performance.  usually 1 sentence each. <br />
 * LONGHIGHLIGHTS paragraph bullet points about company performance <br />
 * PARATEXT paragraph text, i.e., not otherwise formatted <br />
 * TABLE  <br />
 * SECTIONHEADER  <br />
 * ABOUTCOMPANY boilerplate text about the company, such as founding information and slogans <br />
 * LEGALESE text included for legal reasons, often to counterbalance HIGHLIGHTS <br />
 * FOOTNOTE  <br />
 * QUOTE stand-alone text of a quote, often from a high-ranking corporate officer <br />
 * SPEAKER the speaker of a QUOTE <br />
 * 
 * The goal is to classify different parts of the Earnings Summary into different types of text, so that
 * information extraction can be more accurately tuned to the type of text.
 */
public class RunEarningsSummarySentenceClassifier
    implements Constants
{
	// any preprocessing or feature extractors with multiple language options should use English
    public static final String LANGUAGE_CODE = "en";

    // number of cross-validation folds
    public static final int NUM_FOLDS = 3;

    // if case of multi-label classification
    public static final String BIPARTITION_THRESHOLD = "0.5";

    // Note: CV only uses corpusFilePathTrain, dividing it as appropriate.
    public static final String corpusFilePathTrain = "src/test/resources/Annotations/";
    public static final String corpusFilePathTest = "src/test/resources/Annotations/";

    /**
     * Runs extire experiment and prints out results.
     * 
     * @param args
     * @throws Exception
     */
    public static void main(String[] args)
        throws Exception
    {
    	
    	// This is used to ensure that the required DKPRO_HOME environment variable is set.
    	// Ensures that people can run the experiments even if they haven't used DKPro before.
    	// Don't use this in real experiments! Read the documentation and set DKPRO_HOME as explained there.
    	DemoUtils.setDkproHome(PrintTextForAnnotation.class.getSimpleName());
    	
        ParameterSpace pSpace = getParameterSpace();
        
        ExperimentUtils.emptyRepository(); 
        RunEarningsSummarySentenceClassifier experiment = new RunEarningsSummarySentenceClassifier();
        experiment.runCrossValidation(pSpace);
        ResultsMetricsPrinter.callFromGroovyStarter("IeTextClassificationCV"); //print results
    }

    @SuppressWarnings("unchecked")
    public static ParameterSpace getParameterSpace()
    {
        // Train/test will use both reader dimensions, while cross-validation will only use the train part
        Map<String, Object> dimReaders = new HashMap<String, Object>();
        dimReaders.put(DIM_READER_TRAIN, IeSentencePdfReader.class);
        dimReaders.put(
                        DIM_READER_TRAIN_PARAMS,
                        Arrays.asList(IeSentencePdfReader.PARAM_SOURCE_LOCATION,
                                corpusFilePathTrain,
                                IeSentencePdfReader.PARAM_LANGUAGE, LANGUAGE_CODE,
                                IeSentencePdfReader.PARAM_PATTERNS,
                                IeSentencePdfReader.INCLUDE_PREFIX
                                        + "*.txt"));
        dimReaders.put(DIM_READER_TEST, IeSentencePdfReader.class);
        dimReaders.put(
                		DIM_READER_TEST_PARAMS,
                		Arrays.asList(IeSentencePdfReader.PARAM_SOURCE_LOCATION,
                				corpusFilePathTest, 
                				IeSentencePdfReader.PARAM_LANGUAGE, LANGUAGE_CODE, 
                				IeSentencePdfReader.PARAM_PATTERNS, IeSentencePdfReader.INCLUDE_PREFIX 
                				+ "*.txt"));

        // Set the machine learner if using Weka
//        Dimension<List<String>> dimClassificationArgs = Dimension.create(DIM_CLASSIFICATION_ARGS,
//                Arrays.asList(new String[] { SMO.class.getName() }));

        // Configure feature parameters
        Dimension<List<Object>> dimPipelineParameters = Dimension.create(
                DIM_PIPELINE_PARAMS,
                Arrays.asList(new Object[] {
                		NGramFeatureExtractorBase.PARAM_NGRAM_USE_TOP_K, 1000,
                		NGramFeatureExtractorBase.PARAM_NGRAM_MIN_N, 1,
                		NGramFeatureExtractorBase.PARAM_NGRAM_MAX_N, 2,
                		LongWordsFeatureExtractor.PARAM_MIN_CHARS, 3,
                		LongWordsFeatureExtractor.PARAM_MAX_CHARS, 7,
                		}));

        // Set the feature extractors
        Dimension<List<String>> dimFeatureSets = Dimension.create(
                DIM_FEATURE_SET,
                Arrays.asList(new String[] {
                		NrOfTokensUFE.class.getName(),
                		NrOfTokensPerSentenceUFE.class.getName(),
                		LongWordsFeatureExtractor.class.getName(),
                		PastVsFutureFeatureExtractor.class.getName(),
                		POSRatioFeatureExtractor.class.getName(),
                		SuperlativeRatioFeatureExtractor.class.getName(),
                		LuceneNGramUFE.class.getName(),
                		// there's many more, or write your own...
                }
        ));
        
        // if comparing against baseline, set the baseline ML
        Dimension<List<String>> dimBaselineClassificationArgs = Dimension.create(DIM_BASELINE_CLASSIFICATION_ARGS,
        		Arrays.asList(new String[]{NaiveBayes.class.getName()}));
        
        // baseline feature extractors
        Dimension<List<String>> dimBaselinePipelineParameters = Dimension.create(DIM_BASELINE_FEATURE_SET,
        		Arrays.asList(new String[]{
        				NrOfTokensDFE.class.getName(),
        				LuceneNGramDFE.class.getName()
        				}));

        // baseline feature parameters
        Dimension<List<Object>> dimBaselineFeatureSets = Dimension.create(DIM_BASELINE_PIPELINE_PARAMS,
        		Arrays.asList(new Object[]{
        				NGramFeatureExtractorBase.PARAM_NGRAM_USE_TOP_K, 500,
                		NGramFeatureExtractorBase.PARAM_NGRAM_MIN_N, 1,
                        NGramFeatureExtractorBase.PARAM_NGRAM_MAX_N, 3}));

        // we are running a single-label, sequential learning task
        ParameterSpace pSpace = new ParameterSpace(
        		Dimension.createBundle("readers", dimReaders),
                Dimension.create(DIM_LEARNING_MODE, LM_SINGLE_LABEL), 
                Dimension.create(DIM_FEATURE_MODE, FM_SEQUENCE), 
//                Dimension.create(DIM_BIPARTITION_THRESHOLD, BIPARTITION_THRESHOLD), //for multi-label
                dimPipelineParameters, 
                dimFeatureSets,
//                dimClassificationArgs, // if using Weka
                dimBaselineClassificationArgs, 
                dimBaselineFeatureSets, 
                dimBaselinePipelineParameters
                );

        return pSpace;
    }

    // ##### CV #####
    protected void runCrossValidation(ParameterSpace pSpace)
        throws Exception
    {

        ExperimentCrossValidation batch = new ExperimentCrossValidation("IeTextClassificationCV", CRFSuiteAdapter.class,
                NUM_FOLDS); //was CRFSuiteAdapter WekaClassificationAdapter
        // add a second report to TestTask which creates a report about average feature values for
        // each outcome label
        batch.addInnerReport(CRFSuiteClassificationReport.class); //was WekaFeatureValuesReport
        batch.setPreprocessing(getPreprocessing());
        batch.setParameterSpace(pSpace);
        batch.setExecutionPolicy(ExecutionPolicy.RUN_AGAIN);
        batch.addReport(CRFSuiteBatchCrossValidationReport.class); //was BatchCrossValidationReport

        // Run
        Lab.getInstance().run(batch);
    }
    

    // Preprocess our corpus (add POS tags, etc)
    protected AnalysisEngineDescription getPreprocessing()
        throws ResourceInitializationException
    {

        return createEngineDescription(
                createEngineDescription(BreakIteratorSegmenter.class),
                createEngineDescription(OpenNlpPosTagger.class, OpenNlpPosTagger.PARAM_LANGUAGE,
                        LANGUAGE_CODE));
    }
}
