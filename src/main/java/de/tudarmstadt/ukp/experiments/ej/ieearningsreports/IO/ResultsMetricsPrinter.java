package de.tudarmstadt.ukp.experiments.ej.ieearningsreports.IO;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;

import de.tudarmstadt.ukp.dkpro.core.api.resources.DkproContext;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.Utils.ExperimentUtils;
import de.tudarmstadt.ukp.experiments.ej.ieearningsreports.Utils.StatsUtils;

/**
 * To use: 2 methods.
 * With either method:
 * - clear your dkpro workspace
 * 
 * 1) Call from the java GroovyStarter class. Add:
 * ResultsMetricsPrinter.callFromGroovyStarter(String myExperimentName)
 *    or
 * ResultsMetricsPrinter.callFromGroovyStarter(String myExperimentName, iaaFloor, iaaCeil)
 * and get myExperimentName from the groovy file. (Or hardcode it, but this might rarely be wrong).
 * Clear your workspace of this experiment type each time, fresh.
 * 
 * @) Run stand-alone.
 * run any or all of your train/test
 * experiments exactly once.  Make sure each experiment has a unique,
 * useful name.
 * If an experiment breaks halfway, you must go into the workspace and remove the remnants.
 * "rm -r *myexperimentname*"
 * Then, run this class, adding the experiment name in main().  Runtime is very fast.
 * 
 * New feature: Input files instance ID's can have a prefix with IAA,  0-100 + "_".
 * This is "hard cases" versus "easy cases".  
 * 
 * @author jamison
 *
 */
public class ResultsMetricsPrinter
{
	public String experimentName; // such as StemsExperiment
	public boolean useFloorCeiling;
	public int iAACeil; // on a scale of 0-100, as marked in the instanceID
	public int iAAFloor; // like 30-60
	public File repository;
	public List<File> experiments;
	public int numInstances; //counted ++, but should be the same as all fields of totalMatrix, summed
	public List<String> categoryNames;
	public int[][] totalMatrix; //int[golds][preds]
	
	public Double accuracyTotal;
	public Double accuracyAvgPerClass;
	
	public boolean isCRF;
	

	public void setExperimentName(String aName){
		experimentName = aName;
	}
	public void setIaa(int aIaaFloor, int aIaaCeil){
		iAACeil = aIaaCeil;
		iAAFloor = aIaaFloor;
	}
	public void setUseFloorCeiling(boolean aUseFloorCeiling){
		useFloorCeiling = aUseFloorCeiling;
	}
	
	// makes list of experiments we want
	public void initialize() throws IOException{
		
		String subfolder = "/repository/";
//        String subfolder = "/MTurkTrainDataExps/StemsExperiment_all_ub_scmv/";
		isCRF = false;
		repository = new File(DkproContext.getContext().getWorkspace("de.tudarmstadt.ukp.dkpro.lab").getAbsolutePath() + subfolder);
		experiments = new ArrayList<File>();
		numInstances = 0;
		File[] allexperiments = repository.listFiles();
		
//		System.out.println("Looking for " + "WekaTestTask-" + experimentName);
		for(File file: allexperiments){
			if(file.getName().contains("WekaTestTask-" + experimentName)){
//				System.out.println("Found file: " + file.getName());
				experiments.add(file);
			}
		}
		if(experiments.size() == 0){
			for(File file: allexperiments){
				if(file.getName().contains("CRFSuiteTestTask-" + experimentName)){
//					System.out.println("Found file: " + file.getName());
					experiments.add(file);
				}
			}
			if(experiments.size() > 0){
				isCRF = true;
			}else{
				throw new IOException("Error: Only " + experiments.size() + " TestTask files found.  "
						+ "Please fix and try again.");
			}
		}
		
	}
	public void findCategoryNames() throws IOException{
		categoryNames = new ArrayList<String>();

		if(!isCRF){
			for(File file: experiments){
	//			System.out.println("Starting cats in " + file.getName());
				File aResultsFile = new File(file, "id2outcome.txt");
				int linecounter = 0;
				for(String line: FileUtils.readLines(aResultsFile)){
					if(line.startsWith("#")){
						continue;
					}
					String prediction = line.split("=")[line.split("=").length-1].split(";")[0];//[1][0]
					String gold = line.split("=")[line.split("=").length-1].split(";")[1].replace("\n", "");//[1][1]
	//				System.out.println(linecounter + " Prediction: " + prediction);
					if(!categoryNames.contains(prediction)){
						categoryNames.add(prediction);
					}
					if(!categoryNames.contains(gold)){
						categoryNames.add(gold);
					}
					linecounter = linecounter + 1;
				}
			}
		}else{ //isCRF
			for(File file: experiments){
				File aResultsFile = new File(file, "id2outcome.txt");
				for(String line: FileUtils.readLines(aResultsFile)){
					if(line.startsWith("#labels")){
						for(String pair: line.split(" ")){
							if(!pair.contains("=")){
								continue;
							}
							String label = "EMPTY";
							if(pair.split("=").length == 2){
								label = pair.split("=")[1].replace("\n", "");

								if(!categoryNames.contains(label)){//I checked, there's no empties!
									categoryNames.add(label);
								}
							}
//							System.out.println(pair);
						}
						break;
					}
				}
			}
		}
		
		java.util.Collections.sort(categoryNames);
		
//		for(String name: categoryNames){
//			System.out.println("Categoryname: " + name);
//		}
		
		//*Now* we know how big to make the matrix

		totalMatrix = new int[categoryNames.size()][categoryNames.size()];
	}

	public void compileMatrix() throws IOException{

		for(File file: experiments){
			File aResultsFile = new File(file, "id2outcome.txt");
			if(!isCRF){
				addResultsToMatrixWeka(aResultsFile);
			}else{
				addResultsToMatrixCRF(aResultsFile);
			}
		}
	}

	private void addResultsToMatrixCRF(File aResultsFile) throws IOException{
		Map<String, String> labelMap = new HashMap<String, String>(); //num,label
		for(String line: FileUtils.readLines(aResultsFile)){
			if(line.startsWith("#")){
				if(line.startsWith("#labels")){
					labelMap = getLabelMapping(line);
				}
				continue;
			}
			//			System.out.println("iaa=" + aIaa);
			if(useFloorCeiling){
				int aIaa = new Integer(line.split("unit")[1].split("_")[0]); //expecting DOCID_s139_43_2_unit200_s139u2_RT=1;12 where 200 is the IAA
				if(aIaa < iAAFloor || aIaa > iAACeil){
					continue;
				}
			}
			String predictionNum = line.split("=")[line.split("=").length-1].split(";")[0];
			String goldNum = line.split("=")[line.split("=").length-1].split(";")[1].replace("\n", "");

//			System.out.println("goldNum: " + goldNum);
//			System.out.println("predNum: " + predictionNum);
			String prediction = labelMap.get(predictionNum);
			String gold = labelMap.get(goldNum);
//			System.out.println("gold: " + gold);
//			System.out.println("pred: " + prediction);
			
			int goldpointer = categoryNames.lastIndexOf(gold);
			int predictionpointer = categoryNames.lastIndexOf(prediction);
			
			// Format: int[golds][preds]
			totalMatrix[goldpointer][predictionpointer] = totalMatrix[goldpointer][predictionpointer] + 1;
			numInstances++;
		}
	}
	private void addResultsToMatrixWeka(File aResultsFile) throws IOException{
		for(String line: FileUtils.readLines(aResultsFile)){
			if(line.startsWith("#")){
				continue;
			}
			if(useFloorCeiling){
				int aIaa = new Integer(line.split("_")[0]); //expecting 75_1929_VIEW1_1929_VIEW2=0;1 where 75 is the IAA
				if(aIaa < iAAFloor || aIaa > iAACeil){
					continue;
				}
			}
			String prediction = line.split("=")[line.split("=").length-1].split(";")[0];
			String gold = line.split("=")[line.split("=").length-1].split(";")[1].replace("\n", "");
			int goldpointer = categoryNames.lastIndexOf(gold);
			int predictionpointer = categoryNames.lastIndexOf(prediction);
			
			// Format: int[golds][preds]
			totalMatrix[goldpointer][predictionpointer] = totalMatrix[goldpointer][predictionpointer] + 1;
			numInstances++;
		}
	}
	
	public Map<String, String> getLabelMapping(String line){
		Map<String, String> labelMap = new HashMap<String, String>();
		for(String pair: line.split(" ")){
			if(!pair.contains("=")){
				continue;
			}
			String label = "EMPTY"; //I checked, there's no natural empties!
			String num = pair.split("=")[0];
			if(pair.split("=").length == 2){
				label = pair.split("=")[1].replace("\n", "");
				labelMap.put(num, label);
			}
		}
		return labelMap;
	}
	public void calculateTotalAccuracy(){
////		accuracyTotal = new Double(0);
//		double sumCorrect = 0;
//		double sumTotal = 0;
//		for(int i=0;i<categoryNames.size();i++){
//			sumCorrect = sumCorrect + totalMatrix[i][i]; //we want the diagonal!
//			for(int j: totalMatrix[i]){
//				sumTotal = sumTotal + j;
//			}
//		}
//		accuracyTotal = new Double(sumCorrect) / sumTotal;
		accuracyTotal = StatsUtils.getAccuracyMicroAveraged(totalMatrix);
	}
	public void calculateAvgAccuracyPerClass(){
//		accuracyAvgPerClass = new Double(0);
		List<Double> accuraciesPerClass = new ArrayList<Double>();
		for(int i=0;i<totalMatrix.length;i++){
			double sumForAClass = 0;
			for(int j=0;j<totalMatrix[i].length;j++){
				sumForAClass = sumForAClass + totalMatrix[i][j];
			}
			double accForAClass = totalMatrix[i][i] / sumForAClass;
			accuraciesPerClass.add(accForAClass);
		}
		double sumOfAccuracies = 0;
		for(int i=0;i<accuraciesPerClass.size();i++){
			Double acc = accuraciesPerClass.get(i);
			System.out.println(categoryNames.get(i) + " acc: " + acc);
			sumOfAccuracies = sumOfAccuracies + acc;
		}
		accuracyAvgPerClass = sumOfAccuracies / accuraciesPerClass.size();
	}
	//gets the list number for this label/class
	private int getPointer(String label){
		int pointer = categoryNames.lastIndexOf(label);
		return pointer;
	}
    private static Double r4(Double value){
        Double roundedValue = (double)Math.round(value * 10000) / 10000;
        return roundedValue;
    }
	
	@Override
	public String toString(){
		System.out.print("\nHere are the categories: ");
		for(String cat: categoryNames){
			System.out.print(cat + " ");
		}
		System.out.println("\n Note: Categories in evaluation that were unseen in training may be marked as null.\n");
		
		String matrixString = "";
		for(int i=0;i<categoryNames.size();i++){
			matrixString = matrixString + "\t" + categoryNames.get(i).substring(0, 3);
		}
		matrixString = matrixString + " = predictions " + "\n";
		
		for(int i=0;i<totalMatrix.length;i++){
			matrixString = matrixString + categoryNames.get(i).substring(0, 3);
			for(int j=0;j<totalMatrix.length;j++){
				matrixString = matrixString + "\t" + totalMatrix[i][j];
			}
			matrixString = matrixString + "\n";
		}
		matrixString = matrixString + "= golds \n";
		return matrixString;
	} 
	public void printResultsPerClass(String label){
		System.out.println("----Results for label " + label + "----");
		int pointer = getPointer(label);
		Double precision = StatsUtils.getPrecision(
				StatsUtils.getTP(pointer, totalMatrix), StatsUtils.getFP(pointer, totalMatrix));
		Double recall = StatsUtils.getRecall(
				StatsUtils.getTP(pointer, totalMatrix), StatsUtils.getFN(pointer, totalMatrix));
		Double fmeasure = StatsUtils.getF1(precision, recall);
		System.out.println("Precision(" + label + "):" + r4(precision));
		System.out.println("Recall(" + label + "):" + r4(recall));
		System.out.println("Fmeasure(" + label + "):" + r4(fmeasure));
		System.out.println("");
		
	}
	public void printResults(){
		System.out.println("---------CombineTestResults, Classification--------");
		for(String label: categoryNames){
			printResultsPerClass(label);
		}
		Double accuracyMicro = StatsUtils.getAccuracyMicroAveraged(totalMatrix);
		Double fmeasureMicro = StatsUtils.getFMeasureMicroAveraged(totalMatrix);
		Double fmeasureMacro = StatsUtils.getFMeasureMacroAveraged(totalMatrix);
//		Double approxMfcBaseline = StatsUtils.getApproxMostFreqClassBaseline(totalMatrix);
		System.out.println("Total avg micro acc for " + experimentName + ": " + r4(accuracyMicro));
		System.out.println("Micro-avg F-measure: " + r4(fmeasureMicro));
		System.out.println("Macro-avg F-measure: " + r4(fmeasureMacro));
//		System.out.println("Approximate MFC Baseline: " + r4(approxMfcBaseline));
		System.out.println("Num instances: " + numInstances);
		if(useFloorCeiling){
			System.out.println("Test IAA Range (floor, ceil): " + iAAFloor + "--" + iAACeil);
		}
		
	}
	public void printResultsBrief(){
		Double fmeasureMicro = StatsUtils.getFMeasureMicroAveraged(totalMatrix);
		Double fmeasureMacro = StatsUtils.getFMeasureMacroAveraged(totalMatrix);
		System.out.println("MacroFm & MicroFm");
		System.out.println(r4(fmeasureMacro) + " & " + r4(fmeasureMicro));
		ExperimentUtils.printTodaysResults(r4(fmeasureMacro) + " & " + r4(fmeasureMicro));
	}
	public static void callFromGroovyStarter(String myExperimentName, int iAAFloor, int iAACeil) throws IOException{
		ResultsMetricsPrinter combo = new ResultsMetricsPrinter();
		combo.setExperimentName(myExperimentName);
		combo.setUseFloorCeiling(true);
		combo.setIaa(iAAFloor, iAACeil);
		combo.run();
	}
	public static void callFromGroovyStarter(String myExperimentName) throws IOException{
		ResultsMetricsPrinter combo = new ResultsMetricsPrinter();
		combo.setExperimentName(myExperimentName);
		combo.setUseFloorCeiling(false);
		combo.run();
	}
	public void run()throws IOException{
		initialize();
		findCategoryNames();
		compileMatrix();
		calculateTotalAccuracy();
		printResults();
		System.out.println(this.toString()); //prints matrix
//		printResultsBrief(); //for latex
	}
	
	public static void main(String[] args) throws IOException{
//		String myExperimentName = "PosUnitExperiment";
		String myExperimentName = "StemsExperiment";
//		String myExperimentName = "RteOriginalExperiment";
		ResultsMetricsPrinter combo = new ResultsMetricsPrinter();
		combo.setExperimentName(myExperimentName);
		combo.setIaa(30,60);//floor then ceil
		combo.run();
		System.out.println(combo.toString());
	}

}
