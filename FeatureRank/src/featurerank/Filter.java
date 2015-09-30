/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package featurerank;
import java.util.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.AttributeSelection;
import weka.core.Instances;
import org.apache.commons.math3.*;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.summary.Sum;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.filters.unsupervised.attribute.Remove;
//import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.core.WekaPackageManager;
import weka.filters.unsupervised.attribute.MakeIndicator;



class FGene1 {

    int index;
    double value;

    FGene1(int i, double v) {
        this.index = i;
        this.value = v;
    }
}

class FSample {

    int index;
    double value;

    FSample(int i, double v) {
        this.index = i;
        this.value = v;
    }
}

class CompGene implements Comparator<FGene1>//compare in ascending order
{

    @Override
    public int compare(FGene1 a, FGene1 b) {

        if (a.value > b.value) {
            return 1;
        } else if (a.value < b.value) {
            return -1;
        } else {
            return 0;
        }
    }

}

class CompGeneDesc implements Comparator<FGene1> {

    @Override
    public int compare(FGene1 a, FGene1 b) {

        if (a.value > b.value) {
            return -1;
        } else if (a.value < b.value) {
            return 1;
        } else {
            return 0;
        }
    }

}

class CompDoubleAsc implements Comparator<FGene1> {

    @Override
    public int compare(FGene1 a, FGene1 b) {
        if (a.value > b.value) {
            return 1;
        } else if (a.value < b.value) {
            return -1;
        } else {
            return 0;
        }
    }
}

class CompareSample implements Comparator<FSample> {

    @Override
    public int compare(FSample a, FSample b) {

        if (a.value > b.value) {
            return 1;
        } else if (a.value < b.value) {
            return -1;
        } else {
            return 0;
        }

    }

}

class Filter {

    static Instances useFilter(Instances data, Remove rm) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    
    int maxReturnValue; //maximum number of attributes to return
    List<Integer> classList;
    private String[] args;

    Filter(int m) {
        this.maxReturnValue = m;
        this.classList = new ArrayList<>();
    }

    private int findMaxIndex(double[] a) {
        int m = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > a[m]) {
                m = i;
            }
        }
        return m;
    }

    List<Integer> getClassList() throws NullPointerException {
        if (this.classList.size() > 0) {
            return this.classList;
        } else {
            throw new NullPointerException();
        }
    }

    private int findMaxIndex(int[] a) {
        int m = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > a[m]) {
                m = i;
            }
        }
        return m;
    }

    FGene1[][] getRanks(final Instances data,int filterNum) throws Exception {
        switch (filterNum) {
            case 1:
                return pMetricRanking(data);
            case 2:
                return tScoreRanking(data);
            case 3:
                return SVMBasedRanking(data);
            case 4:
                return classifierBasedRanking(data);
            
               
            default:
                throw new Exception("Invalid Filter name");
        }
    }
   

   /* Integer[] getFeatures(final Instances data, final FGene1[][] ranks, String subSelectionName, String classifierName) throws Exception {

        switch (subSelectionName) {
            case "AvgRank":
                return maxRelevantAvgRank(data, ranks, classifierName);
            case "MinRank":
                return maxRelevantAvgMinRank(data, ranks, classifierName);
            case "WD":
                return MaxRelevantWeightedAvg(data, ranks, classifierName);
            case "CADA":
                return MaxRelevantAdpative(data, ranks, classifierName);
            default:
                throw new Exception("Invalid Subset Selection method name");
        }
    }

    private Classifier getClassifier(String classifierName) throws Exception {
        switch (classifierName) {
            case "NB":
                return new NaiveBayes();
            case "SVM":
                return new SMO();
            case "DT":
                return new J48();
            default:
                throw new Exception("Invalid Classifier Name");
        }
    }*/
    void co(Instances data) throws Exception
    {
        AttributeSelection attr = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();  //BestFirst bfs = new BestFirst();                                                                                  
        //breader.close();                                                                                     
        attr.setEvaluator(eval);
        attr.setSearch(search);                                                      //attr.setSearch(bfs);
        attr.SelectAttributes(data);
        System.out.println(attr.toResultsString())  ; 
    }
    Integer[] maxRelevantAvgRank(final Instances data , final FGene1[][] ranks) throws IOException
	{
            File file = new File("F:/shivani/AffinityPropagation/pmetricfile.txt");
                // if file doesnt exists, then create it
                if (!file.exists()) {
                    file.createNewFile();
                    }
 
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            int HighValueGenes = (int) Math.ceil(0.06*(data.numAttributes()));
            int flag = -1;
            System.out.println("no of high value genes "+HighValueGenes);
            int x = 0;
            // selecting class wise top ranked genes 
            for(int aa = 0;aa < data.numAttributes();aa++)
            {
            for(int cc = 0;cc<data.numClasses();cc++)
            {
                
                //String content = (data.attribute(ranks[cc][aa].index).toString().replaceAll("@attribute", "").replaceAll("numeric","").replaceAll("(.*)at", ""));
                String content = Integer.toString(data.attribute(ranks[cc][aa].index).index());
                System.out.println(content);
                flag++;
                if(flag == HighValueGenes)
                {
                    x = 1;
                    break;
                }
                    
               System.out.println("flag "+ flag);
                bw.write(content + " ");
                
            }
            if(x ==1)
                break;
            }
            
          
            System.out.println("Done");
            bw.close(); 
            
//            
//            
            File file1 = new File("F:/shivani/AffinityPropagation/highfile1.txt");
            File file2 = new File("F:/shivani/AffinityPropagation/highfile2.txt");
//            File file3 = new File("F:/shivani/AffinityPropagation/highfile3.txt");
//            
            File file4 = new File("F:/shivani/AffinityPropagation/lowfile1.txt");
            File file5 = new File("F:/shivani/AffinityPropagation/lowfile2.txt");
//            File file6 = new File("F:/shivani/AffinityPropagation/lowfile3.txt");
//           File file7 = new File("F:/shivani/AffinityPropagation/highfile4.txt");
//            File file8 = new File("F:/shivani/AffinityPropagation/lowfile4.txt");
//                // if file doesnt exists, then create it
                if (!file1.exists()) {
                    file1.createNewFile();
                    }
                if (!file2.exists()) {
                    file2.createNewFile();
                    }
////                
                if (!file4.exists()) {
                    file4.createNewFile();
                    }
                if (!file5.exists()) {
                    file5.createNewFile();
                    }
////                
//                if (!file7.exists()) {
//                    file7.createNewFile();
//                    }
//                if (!file8.exists()) {
//                    file8.createNewFile();
//                    }
//// 
// 
            FileWriter fw1 = new FileWriter(file1.getAbsoluteFile());
            FileWriter fw2 = new FileWriter(file2.getAbsoluteFile());
//            FileWriter fw3 = new FileWriter(file3.getAbsoluteFile());
            FileWriter fw4 = new FileWriter(file4.getAbsoluteFile());
            FileWriter fw5 = new FileWriter(file5.getAbsoluteFile());
//           FileWriter fw6 = new FileWriter(file6.getAbsoluteFile());
//            FileWriter fw7 = new FileWriter(file7.getAbsoluteFile());
//          FileWriter fw8 = new FileWriter(file8.getAbsoluteFile());
            BufferedWriter bw1 = new BufferedWriter(fw1);
            BufferedWriter bw2 = new BufferedWriter(fw2);
//            BufferedWriter bw3 = new BufferedWriter(fw3);
            BufferedWriter bw4 = new BufferedWriter(fw4);
            BufferedWriter bw5 = new BufferedWriter(fw5);
//           BufferedWriter bw6 = new BufferedWriter(fw6);
//            BufferedWriter bw7 = new BufferedWriter(fw7);
//           BufferedWriter bw8 = new BufferedWriter(fw8);
            //int High = (int) Math.ceil(0.50*(data.numAttributes()));
          //  int flag = -1;
           // System.out.println("no of high value genes "+High);
            //int x = 0;
            int part = data.numAttributes()/2;
               for(int aa = 0;aa< part;aa++)
                {
                    String content_high1 = Integer.toString(data.attribute(ranks[0][aa].index).index());
                    bw1.write(content_high1 + " " );
                    
                }
               bw1.close();
               for(int aa = 0;aa< part;aa++)
                {
                    String content_high2 = Integer.toString(data.attribute(ranks[1][aa].index).index());
                    bw2.write(content_high2 + " ");
                    
                }
               bw2.close();
//               for(int aa = 0;aa< part;aa++)
//                {
//                    String content_high3 = Integer.toString(data.attribute(ranks[2][aa].index).index());
//                    bw3.write(content_high3 + " ");
//                    
//                }
//               bw3.close();
               for(int aa = part;aa< data.numAttributes() -1;aa++)
                {
                    String content_low1 = Integer.toString(data.attribute(ranks[0][aa].index).index());
                    bw4.write(content_low1 + " ");
                    
                }
               bw4.close();
               for(int aa = part;aa< data.numAttributes() -1;aa++)
                {
                    String content_low2 = Integer.toString(data.attribute(ranks[1][aa].index).index());
                    bw5.write(content_low2 + " ");
                    
                    
                }
            bw5.close();
//            for(int aa = part;aa< data.numAttributes() -1;aa++)
//                {
//                    String content_low3 = Integer.toString(data.attribute(ranks[2][aa].index).index());
//                    bw6.write(content_low3 + " ");
//                    
//                    
//                }
//            bw6.close();
//          for(int aa = 0;aa< part ;aa++)
//                {
//                    String content_low3 = Integer.toString(data.attribute(ranks[3][aa].index).index());
//                    bw7.write(content_low3 + " ");
//                    
//                    
//                }
//            bw7.close();
//            for(int aa = part;aa< data.numAttributes() -1;aa++)
//                {
//                    String content_low3 = Integer.toString(data.attribute(ranks[3][aa].index).index());
//                    bw8.write(content_low3 + " ");
//                    
//                    
//                }
//            bw8.close();
//          
            for(int cc = 0;cc<data.numClasses();cc++)
            {
                for(int aa = 0;aa < data.numAttributes() -1;aa++)
                {
                    
                System.out.println(ranks[cc][aa].index+"     "+ranks[cc][aa].value );
                }
                System.out.println("over");
                
            }
            
            Integer[] retArr = new Integer[this.maxReturnValue];
            List<HashMap<Integer,FGene1>> lhm = new ArrayList<HashMap<Integer,FGene1>>();
            try
		{
	         //read ranks from file and create a HashMap as <index,FGene>	FGene--> index:rank , value:fmeasure 
                    for(int i = 0 ; i < data.numClasses() ; i++)
			{
                            HashMap<Integer,FGene1> hm = new HashMap<Integer, FGene1>();
                            for(int xx = 0 ; xx < data.numAttributes() - 1 ; xx++)
				{
                                 //  System.out.println("class no  "+ i + " attribute no "+ xx+" "+data.attribute(ranks[i][xx].index) + " index " + ranks[i][xx].index + " value "+ranks[i][xx].value);
                                    int RankAsIndex = xx+1; // rank of gene from a sorted list
                                    //System.out.println(i+" "+xx);
                                    int index = ranks[i][xx].index; // actual index of gene
                                    double tempValue  = ranks[i][xx].value;
                                    FGene1 temp  = new FGene1(RankAsIndex , tempValue);//FGene--> index = rank , value = fmeasure
                                    hm.put(index,temp);
				}
                            lhm.add(hm);
			}
                     List<FGene1> tempList = new ArrayList<FGene1>();
                     //calculate avg rank for each gene 
                    for(int i = 0 ; i < data.numAttributes() -1 ; i++)
                        {
                            double avg = 0.0;
                            int tempCount = 0;
                            for(int j = 0 ; j < data.numClasses()  ; j++) // include class + overall
                                {
                                    if(lhm.get(j).get(i).value > 0.0)//if fmeasure is zero do not include gene rank in average
                                        {
                                      //  System.out.println("i " +i+ " j " +j+" index " +lhm.get(j).get(i).index);
                                            avg += lhm.get(j).get(i).index;
                                            tempCount++;
                                        }
                                }
                                    tempList.add(new FGene1(i , avg/tempCount)); //FGene--> index:actual index of attribute , value: avg rank of attribute
                        }
                    Collections.sort(tempList,new CompDoubleAsc());//smallest rank at top
                    
                    for( int i = 0 ; i < this.maxReturnValue-1 ; i++)
                        {
                            
                         //  System.out.println("temp list "+ "i "+ i+ " "+tempList.get(i).index+" "+tempList.get(i).value);
                            retArr[i] = tempList.get(i).index;
                           
                        }
                  
                }
            
            catch(Exception e){ e.printStackTrace();}	
		return retArr;
	}
    FGene1[][] pMetricRanking(final Instances data) throws Exception {
        
            
        FGene1[][] rankArr = new FGene1[data.numClasses()][data.numAttributes() - 1];
        try {
            for (int classVal = 0; classVal < data.numClasses(); classVal++) {
            List<FGene1> l1 = new ArrayList<>();//list to hold metric for all genes
            double pMetric;
            for (int numAtt = 0; numAtt < data.numAttributes() - 1; numAtt++) 
                {
                    //for each gene, calculate p metric on one vs all (class samples) basis
                    List<Double> classlist = new ArrayList<>();
                    List<Double> nonClassList = new ArrayList<>();
                    double classMean, nonClassMean, classStd, nonClassStd;
                    for (int numSmpl = 0; numSmpl < data.numInstances(); numSmpl++) 
                    {

                        if ((int) data.instance(numSmpl).classValue() == classVal)
                        {
                            classlist.add(data.instance(numSmpl).value(numAtt));
                        } else
                        {
                            nonClassList.add(data.instance(numSmpl).value(numAtt));
                        }
                    }

                    Double[] classArr = classlist.toArray(new Double[classlist.size()]);
                    Double[] nonClassArr = nonClassList.toArray(new Double[nonClassList.size()]);
                    double[] classArray = new double[classArr.length];
                    double[] nonClassArray = new double[nonClassArr.length];

                    for (int arrIndex = 0; arrIndex < classArray.length; arrIndex++) //wrapper to native type
                    {
                        classArray[arrIndex] = classArr[arrIndex];
                    }

                    for (int arrIndex = 0; arrIndex < nonClassArray.length; arrIndex++)//wrapper to native type
                    {
                        nonClassArray[arrIndex] = nonClassArr[arrIndex];
                    }

                    Mean mean = new Mean();
                    classMean = mean.evaluate(classArray);
                    nonClassMean = mean.evaluate(nonClassArray);
                    StandardDeviation sd = new StandardDeviation();
                    classStd = sd.evaluate(classArray);
                    nonClassStd = sd.evaluate(nonClassArray);
                    //					pMetric = Math.pow((classMean - nonClassMean),2)/(Math.pow(classStd , 2) + Math.pow(nonClassStd , 2));
                    pMetric = Math.abs(classMean - nonClassMean) / (classStd + nonClassStd);

                    //	System.out.println(pMetric);
                    l1.add(new FGene1(numAtt, pMetric));
                   
                    System.out.println(numAtt + " "+ pMetric);
                    
                }
                 

                Collections.sort(l1, new CompGeneDesc());
               // System.out.println("check" + l1.get(1));//sort in descending on the basis on p value
                FGene1[] sortedFArray1 = l1.toArray(new FGene1[l1.size()]);
                System.arraycopy(sortedFArray1, 0, rankArr[classVal], 0, data.numAttributes() - 1); 
               // for (int i = 0; i < data.numAttributes() - 1; i++) {
                //rankArr[classVal][i] = sortedFArray1[i];  // manual array copy
                //System.out.println(rankArr[1][2].index);
                //     System.out.println ("gene" + data.attribute(rankArr[1][2].index));
                //System.out.println("VALUE:"+rankArr[classVal][i].value);
            }
        } 
        catch (Exception e) 
        {
        }
        
        //MaxRelevantAdpative(data , rankArr);
       Integer[] i = maxRelevantAvgRank(data, rankArr);
       System.out.println("P metric ranking");
        File file = new File("F:/shivani/AffinityPropagation/newpmetricfile.txt");
                // if file doesnt exists, then create it
                if (!file.exists()) {
                    file.createNewFile();
                    }
 
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            
            int x = 0;
            // selecting class wise top ranked genes 
           for(int p = 0;p <data.numAttributes() -1;p++)
           {
                //String content = (data.attribute(ranks[cc][aa].index).toString().replaceAll("@attribute", "").replaceAll("numeric","").replaceAll("(.*)at", ""));
                String content = Integer.toString(i[p]);
                System.out.println(content);
               
                bw.write(content + " ");
                
            }
            
            
            
            System.out.println("Done");
            bw.close(); 
            
            
       
       
       
       //System.out.println("length "+i.length);
       for(int p = 0;p <data.numAttributes() -1;p++)
       System.out.println("Index " + i[p]+" Attribute name " +data.attribute(i[p].intValue()).name()); // final pmetric rank list
        // maxRelevantAvgMinRank(data, rankArr);
       
       // Checking accuracy with pmteric ranked attributes
      
//     Instances       instNew;
//     Remove          remove;
//     
//     
//      int[] remArray = new int[12];
//     remArray[11] = data.numAttributes()-1;
//     
//     remove = new Remove();
//     remove.setAttributeIndicesArray(remArray);
//     
//     remove.setInvertSelection(true);
//     remove.setInputFormat(data);
//     
//     instNew = weka.filters.Filter.useFilter(data, remove);
//     instNew.setClassIndex(instNew.numAttributes()-1);
//    
//    data.setClassIndex(data.numAttributes()-1);
//    
//     Classifier cls = new J48();
//     SMO sm = new SMO();
//     NaiveBayes nb = new NaiveBayes();
//     Evaluation eval = new Evaluation(instNew);
//     Debug.Random rand = new Debug.Random(1);  // using seed = 1
//     int folds = 10;
//     eval.crossValidateModel(sm, instNew, folds, rand);
//  // eval.crossValidateModel(cls, data, folds, rand);
//     System.out.println(eval.toSummaryString());

        return rankArr;
    }
   

    FGene1[][] SVMBasedRanking(final Instances data1) {
        FGene1[][] rankArr = new FGene1[data1.numClasses()][data1.numAttributes() - 1];

        try {
            for (int i = 0; i < data1.numClasses(); i++) {
                Instances data = new Instances(data1);
                BitSet bs = new BitSet(data.numAttributes() - 1);

                List<FGene1> l1 = new ArrayList<>();//list to hold metric for all genes

                for (int x = 0; x < data.numInstances(); x++) {
                    if (i != (int) data.instance(x).classValue()) {
                        data.instance(x).setValue(data.classIndex(), 0);
                    } else {
                        data.instance(x).setValue(data.classIndex(), 1);
                    }
                }

//                				MakeIndicator filter = new MakeIndicator();
//                				filter.setAttributeIndex("" + (data.classIndex() + 1));
//                				
//                				filter.setNumeric(false);
//                				filter.setValueIndex(i);
//                				filter.setInputFormat(data);
//                				Instances trainCopy = Filter.useFilter(data, filter);
//                				System.out.println(data);
                				//bw.write(data.toString());
                				//bw.flush();
//                				System.exit(0);
                SMO smo = new SMO();
                smo.buildClassifier(data);
                double[] weightsSparse = smo.sparseWeights()[0][1];
                int[] indicesSparse = smo.sparseIndices()[0][1];
//                double[] weights = new double[data.numAttributes()];
                for (int j = 0; j < weightsSparse.length; j++) {
//                    weights[indicesSparse[j]] = weightsSparse[j] * weightsSparse[j];
                    l1.add(new FGene1(indicesSparse[j], weightsSparse[j] * weightsSparse[j]));
                    bs.set(indicesSparse[j]);
                }
                for (int j = 0; j < data.numAttributes() - 1; j++) {
                    if (!bs.get(j)) {
                        l1.add(new FGene1(j, Double.MIN_VALUE));
                    }
                }
                if (l1.size() != data.numAttributes() - 1) {
                    throw new Exception();
                }

                Collections.sort(l1, new CompGeneDesc());//sort in descending on the basis on p value
                rankArr[i] = l1.toArray(new FGene1[l1.size()]);
            }
        } catch (Exception e) {
        }
        return rankArr;
    }

    FGene1[][] tScoreRanking(final Instances data) {
        FGene1[][] rankArr = new FGene1[data.numClasses()][data.numAttributes() - 1];
        try {
            for (int classVal = 0; classVal < data.numClasses(); classVal++) {
                List<FGene1> l1 = new ArrayList<>();//list to hold metric for all genes
                double tScore = 0.0;

                for (int numAtt = 0; numAtt < data.numAttributes() - 1; numAtt++) {
                  //  System.out.println(classVal + " " + numAtt);
                    //for each gene, calculate p metric on one vs all (class samples) basis
                    List<Double> classlist = new ArrayList<>();
                    List<Double> nonClassList = new ArrayList<>();
                    double classMean, nonClassMean, classStd, nonClassStd;
                    for (int numSmpl = 0; numSmpl < data.numInstances(); numSmpl++) {

                        if ((int) data.instance(numSmpl).classValue() == classVal) {
                            classlist.add(data.instance(numSmpl).value(numAtt));
                        } else {
                            nonClassList.add(data.instance(numSmpl).value(numAtt));
                        }
                    }

                    Double[] classArr = classlist.toArray(new Double[classlist.size()]);
                    Double[] nonClassArr = nonClassList.toArray(new Double[nonClassList.size()]);
                    double[] classArray = new double[classArr.length];
                    double[] nonClassArray = new double[nonClassArr.length];
                    for (int arrIndex = 0; arrIndex < classArray.length; arrIndex++) //wrapper to native type
                    {
                        classArray[arrIndex] = classArr[arrIndex];
                    }

                    for (int arrIndex = 0; arrIndex < nonClassArray.length; arrIndex++)//wrapper to native type
                    {
                        nonClassArray[arrIndex] = nonClassArr[arrIndex];
                    }

                    Mean mean = new Mean();
                    classMean = mean.evaluate(classArray);
                    nonClassMean = mean.evaluate(nonClassArray);
                    StandardDeviation sd = new StandardDeviation();
                    classStd = sd.evaluate(classArray);
                    nonClassStd = sd.evaluate(nonClassArray);
                    tScore = Math.abs(classMean - nonClassMean) / Math.pow((classArray.length * Math.pow(classStd, 2.0) + nonClassArray.length * Math.pow(nonClassStd, 2.0)) / (classArray.length + nonClassArray.length), 0.5);
                    l1.add(new FGene1(numAtt, tScore));
                }
                Collections.sort(l1, new CompGeneDesc());//sort in descending on the basis on t score
                FGene1[] sortedFArray1 = l1.toArray(new FGene1[l1.size()]);
                //for (int i = 0; i < data.numAttributes() - 1; i++) { manual array copy
                  //  rankArr[classVal][i] = sortedFArray1[i];
                System.arraycopy(sortedFArray1, 0, rankArr[classVal], 0, data.numAttributes() - 1);
            }
        } catch (Exception e) {
        }
        return rankArr;
    }

    FGene1[][] classifierBasedRanking(final Instances data) {
        FGene1[][] rankArr = new FGene1[data.numClasses()][data.numAttributes() - 1];
        try {

            for (int i = 0; i < data.numAttributes() - 1; i++) {
                //System.out.println("Analyzing gene "+i);
                int[] remArray = new int[2];
                remArray[0] = i;
                remArray[1] = data.classIndex();
                Remove rm = new Remove();
                rm.setAttributeIndicesArray(remArray);
                rm.setInvertSelection(true);//keep the selected indices ..delete all others
                rm.setInputFormat(data);
                //Instances trainingSet = new Instances(data);
               Instances trainingSet = Filter.useFilter(data, rm);
                trainingSet.setClassIndex(trainingSet.numAttributes() - 1);

                Evaluation eval = new Evaluation  (trainingSet);
                //			NaiveBayes sm = new NaiveBayes();
                SMO sm = new SMO();
                eval.crossValidateModel(sm, trainingSet, 10, new Random(1));
                System.out.println(eval.toSummaryString());
                //store results:classwise and overall
                for (int j = 0; j < data.numClasses(); j++) {
                    rankArr[j][i] = new FGene1(i, eval.fMeasure(j));
                }

            }	//end of for		

            //sort indicies based on fmeasures
            for (int j = 0; j < data.numClasses(); j++) {
                java.util.Arrays.sort(rankArr[j], new CompGeneDesc());
            }
        } catch (Exception e) {
        }
        return rankArr;
        //MaxRelevantAdpative(data , rankArr);

    }
}