package featurerank;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.PatternSyntaxException;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.AttributeSelection;
import weka.core.Instances;
import java.util.Scanner;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.summary.Sum;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.AttributeSelection;
import weka.core.WekaPackageManager;
import org.pentaho.packageManagement.PackageManager;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.Remove;



/**
 *
 * @author shivani
 */
public class FeatureRank {

   static void Fscore(final Instances data1)
    {
        List<Double> pv = new ArrayList<>();
        List<Double> num = new ArrayList<>();
        List<Double> numall = new ArrayList<>();
        List<Double> numerator = new ArrayList<>();
        double pooledvar,classStd,kclassMean,classMean,num2,num1;
        double den2=data1.numInstances()-data1.numClasses();
        double den1 = data1.numClasses()-1;
          
          for(int numAtt =0;numAtt<data1.numAttributes()-1;numAtt++)
          {
              List<Double> kclassList = new ArrayList<>();
              List<Double> classList = new ArrayList<>();
               for(int numSmpl =0;numSmpl <data1.numInstances();numSmpl++)
               {
                   classList.add(data1.instance(numSmpl).value(numAtt));
               }
              
              for(int classVal =0;classVal <data1.numClasses();classVal++)
               {
              for(int numSmpl =0;numSmpl <data1.numInstances();numSmpl++)
               {
                if ((int) data1.instance(numSmpl).classValue() == classVal) 
                {
                   kclassList.add(data1.instance(numSmpl).value(numAtt));
                } 
                
               }
             
               Double[] kclassArr = classList.toArray(new Double[kclassList.size()]);
               double[] kclassArray = new double[kclassArr.length];
               for (int arrIndex = 0; arrIndex < kclassArray.length; arrIndex++) //wrapper to native type
               {
                    kclassArray[arrIndex] = kclassArr[arrIndex];
                }
               Double[] classArr = classList.toArray(new Double[classList.size()]);
               double[] classArray = new double[classArr.length];
               for (int arrIndex = 0; arrIndex < classArray.length; arrIndex++) //wrapper to native type
               {
                    classArray[arrIndex] = classArr[arrIndex];
                }
               StandardDeviation sd = new StandardDeviation();
               Mean m = new Mean();
               classStd = sd.evaluate(kclassArray);
               kclassMean = m.evaluate(kclassArray);
               classMean = m.evaluate(classArray);
               num.add(( Math.pow(classStd,2))*(kclassArr.length-1));
               numall.add(kclassArr.length*(kclassMean-classMean));
               }
               Double[] classS = num.toArray(new Double[num.size()]);
               double[] classSum = new double[classS.length];
            //  System.out.println("jsja" +num.get(numAtt));
               for (int arrIndex = 0; arrIndex < classS.length; arrIndex++) 
               {
                   classSum[arrIndex] = classS[arrIndex];
               }
               Double[] classS1 = num.toArray(new Double[numall.size()]);
               double[] classSum1 = new double[classS1.length];
            //  System.out.println("jsja" +num.get(numAtt));
               for (int arrIndex = 0; arrIndex < classS1.length; arrIndex++)
               {
                   classSum1[arrIndex] = classS1[arrIndex];
               }
            Sum s = new Sum();  
            num2 =s.evaluate(classSum);
            pooledvar = num2/den2;
            num1 =s.evaluate(classSum1);
           double nume= num1/den1;
            //System.out.println("ns" + num2);
            pv.add(pooledvar);
            numerator.add(nume);
        }
         int n = pv.size();
         for(int i = 0; i < n ; i++)
         System.out.println( numerator.get( i ) + " " +pv.get(i));
        // System.out.println( numerator.get( i )/pv.get(i));
 

        
    }
  static void mutualinf(Instances data)
   {
       double[] secondVector={0};  
       double[] firstVector = {0}  ; 
       for(int i =0;i<data.numAttributes()-1;i++)
       {
          
           for(int k =0;k<data.numInstances();k++)
             {
                 List<Double> first= new ArrayList<>();
                 first.add(data.instance(k).value(i));
                 Double[] firstVec = first.toArray(new Double[first.size()]);
                 firstVector = new double[firstVec.length];
                 for (int arrIndex = 0; arrIndex < firstVector.length; arrIndex++) //wrapper to native type
                  {
                        firstVector[arrIndex] = firstVec[arrIndex];
                  }
             }
          
           for(int j =1;j<data.numAttributes() -1;j++)
           {
             
             for(int k =0;k<data.numInstances();k++)
             {
                 List<Double> second = new ArrayList<>();
                 second.add(data.instance(k).value(j));
                 Double[] secondVec = second.toArray(new Double[second.size()]);
                 secondVector = new double[secondVec.length];
                 for (int arrIndex = 0; arrIndex < secondVector.length; arrIndex++) //wrapper to native type
                  {
                        secondVector[arrIndex] = secondVec[arrIndex];
                  }
                 System.out.println("nx" + firstVector[0] +" "+ secondVector[0]);
             }
           }
       MutualInformation m = null ;
      // double mi = m.calculateMutualInformation(firstVector, secondVector);
       double mi = MutualInformation.calculateMutualInformation(firstVector, secondVector);
      // System.out.println("nx" + firstVector[0] +" "+ secondVector[0]);
           
       }
       
       
   }
    public static void main(String[] args) throws Exception
    {
        
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("F:/shivani/dataset/arff/chen-2002.arff"));
        Instances data = new Instances(breader);
        data.setClassIndex(data.numAttributes() - 1);
        int s;
        for(int j =0;j<data.numAttributes()-1;j++)
            {
               // System.out.println(fg[i][j].index);
                //System.out.println(fg[i][j].value);
              //System.out.println(data.attribute(j).name());
             
                
            }
              
        Filter f = new Filter(data.numAttributes());
        Scanner in = new Scanner(System.in);
        System.out.println("1. pMetric 2. tScore 3.SVM 4.Classification based 5.Corelation Based  6. Information gain 7. F Score 8. mRMR");
        System.out.println("Select filtering metric");
        s = in.nextInt();
        FGene1[][] fg ;
      //  FGene1[][] fg = new FGene1[data.numAttributes()][data.numAttributes()];
        if(s==8)
        {
            mutualinf(data);
            
        }
        
       else if(s==7)
        {
            Fscore(data);
        }
        else if(s== 6)
        {
        InfoGainAttributeEval ig = new InfoGainAttributeEval();
        ig.runEvaluator(ig, args);
        AttributeSelection attr = new AttributeSelection();
        Ranker search1 = new Ranker();  
        attr.setEvaluator(ig);
        attr.setSearch(search1);
        attr.SelectAttributes(data);
       
        System.out.println(attr.toResultsString());
        
        }
        else if(s==5)
        {
        AttributeSelection attr = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();  //BestFirst bfs = new BestFirst();                                                                                  
        breader.close();                                                                                     
        attr.setEvaluator(eval);
        attr.setSearch(search);
        attr.setRanking(true);
      //attr.setSearch(bfs);
        attr.SelectAttributes(data);
//        Random r = new Random();
//        Evaluation e = new Evaluation(data);
//         e.crossValidateModel(eval, data, 10, new Random(1));
        int sa = attr.numberAttributesSelected();
        
        //System.out.println(sa);
        Instances data1 = attr.reduceDimensionality(data);
        data1.enumerateAttributes();
        //System.out.println(attr.toResultsString())  ;     
        int a[] = new int[100];
        a = attr.selectedAttributes();
        for(int k = 0;k<29;k++)
        {
        //System.out.println("selected attribute " + a[k]);
        //System.out.println("new value "+ data1.attribute(k).name());
        }
        for(int h = 0;h<sa;h++)
        System.out.println(data1.attribute(h).name().replaceAll("at(.*)", " "));
       // a = attr.rankedAttributes();
       // System.out.println("ranked value " + a[0][0]);
       
         for(int i = 0;i<data.numClasses();i++)
            {
                for(int j =0;j<data.numAttributes()-1;j++)
                    {
                       
             //  System.out.println(fg[i][j].index);
                //System.out.println(fg[i][j].value);
              // System.out.println(data.attribute(j).name().replaceAll("at(.*)", "at"));
                //System.out.println(data.attribute(fg[i][j].index));
                
                    }
                PrintStream out;
           try {
            out = new PrintStream(new FileOutputStream("GOterm.txt", true));
            for(int j =0;j<100;j++)
                {
           
           //         System.setOut(out);
                } 
               }
            catch (FileNotFoundException e) {
                }
            }
        
        }
        else
        {
            
        fg = f.getRanks(data, s);
      //  for(int k = 0;k<100;k++)
       // {
        int k = 11;
        for(int i = 0;i<data.numClasses();i++)
        {
            for(int j =0;j<data.numAttributes()-1;j++)
            {
//                System.out.print(fg[i][j].index);
//                System.out.println(fg[i][j].value);
//                System.out.println(data.attribute(j).name().replaceAll("at(.*)", "at"));// printing all the values
                if(k == 0)
                    break;
//                System.out.println(data.attribute(fg[i][j].index).name().replaceAll("at(.*)", "at"));
                k--;
            }
            
        

      
            }
        }
          
      
        }
      }
     
