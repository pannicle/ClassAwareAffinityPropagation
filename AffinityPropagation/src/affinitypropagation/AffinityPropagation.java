
package affinitypropagation;

/**
 *
 * @author shivani
 */
//import static org.jocl.CL.*;
import gov.sandia.cognition.algorithm.MeasurablePerformanceAlgorithm;
import gov.sandia.cognition.learning.algorithm.AbstractAnytimeBatchLearner;
import gov.sandia.cognition.learning.algorithm.clustering.cluster.CentroidCluster;
import gov.sandia.cognition.learning.algorithm.clustering.*;
import gov.sandia.cognition.learning.function.distance.DivergenceFunctionContainer;
import gov.sandia.cognition.learning.function.distance.EuclideanDistanceSquaredMetric;
import gov.sandia.cognition.math.DivergenceFunction;
import gov.sandia.cognition.math.matrix.Vectorizable;
//import gov.sandia.cognition.math.matrix.mtj.Vector3;
import gov.sandia.cognition.math.matrix.mtj.DenseVector;
import gov.sandia.cognition.util.DefaultNamedValue;
import gov.sandia.cognition.util.NamedValue;
import gov.sandia.cognition.util.ObjectUtil;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Arrays;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Debug;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RenameAttribute;



class myVector extends DenseVector{
   public myVector(double a[])
   {
       super(a);
   }
}


public final class AffinityPropagation<DataType>
    extends AbstractAnytimeBatchLearner<Collection<? extends DataType>,Collection<CentroidCluster<DataType>>>
    implements BatchClusterer<DataType, CentroidCluster<DataType>>,
        MeasurablePerformanceAlgorithm,
        DivergenceFunctionContainer<DataType, DataType>
{
    
    
    
    public static final int DEFAULT_MAX_ITERATIONS = 100;
    public static final double DEFAULT_SELF_DIVERGENCE = 0.0;
    public static final double DEFAULT_DAMPING_FACTOR = 0.5;
    protected DivergenceFunction<? super DataType, ? super DataType> divergence;
    private double selfDivergence;
    protected double dampingFactor;
    protected double oneMinusDampingFactor;
    protected transient int exampleCount;
    protected ArrayList<DataType> examples;
    protected  double[][] similarities;
    protected  double[][] similarities_n;
    protected static double[][] mat;
    protected double[][] responsibilities;
    protected double[][] availabilities;
    protected int[] assignments;
    protected int changedCount;
    protected double max2 = -1000000;
    int k = 0; 
    protected HashMap<Integer, CentroidCluster<DataType>> clusters;
    public AffinityPropagation()
    {
        this(null, DEFAULT_SELF_DIVERGENCE);
    }
    public AffinityPropagation(
        DivergenceFunction<? super DataType, ? super DataType> divergence,
        double selfDivergence)
    {
        this(divergence, selfDivergence, DEFAULT_DAMPING_FACTOR);
    }
    public AffinityPropagation(
        DivergenceFunction<? super DataType, ? super DataType> divergence,
        double selfDivergence,
        double dampingFactor)
    {
        this(divergence, selfDivergence, dampingFactor, DEFAULT_MAX_ITERATIONS);
    }

    public AffinityPropagation(
        DivergenceFunction<? super DataType, ? super DataType> divergence,
        double selfDivergence,
        double dampingFactor,
        int maxIterations)
    {
        super(maxIterations);

        this.setDivergence(divergence);
        this.setSelfDivergence(selfDivergence);
        this.setDampingFactor(dampingFactor);
    }
    
    @Override
    public AffinityPropagation<DataType> clone()
    {
        @SuppressWarnings("unchecked")
        final AffinityPropagation<DataType> result = 
            (AffinityPropagation<DataType>) super.clone();
        result.divergence = ObjectUtil.cloneSmart(this.divergence);
        result.exampleCount = 0;
        result.examples = null;
        result.similarities = null;
        result.responsibilities = null;
        result.availabilities = null;
        result.assignments = null;
        result.changedCount = 0;
        result.clusters = null;
        
        return result;
    }
   
    
    @Override
    protected boolean initializeAlgorithm() 
    {
        if (this.getData() == null || this.getData().size() <= 0)
        {
            
            return false;
        }
       // exampleCount = 10;
        // Initialize the main data for the algorithm.
        this.setExamples(new ArrayList<>(this.getData()));
        //System.out.println("get data "+ this.getData());
        this.setSimilarities(new double[this.exampleCount][this.exampleCount]);
        this.setResponsibilities(new double[this.exampleCount][this.exampleCount]);
        this.setAvailabilities(new double[this.exampleCount][this.exampleCount]);
//        double sum = 0;
//        double mean;
//        try {
//            //  System.out.println("example count "+this.exampleCount);
//            // similarity matrix.
//            this.calculate();
//        } catch (Exception ex) {
//            Logger.getLogger(AffinityPropagation.class.getName()).log(Level.SEVERE, null, ex);
//        }
        
        for (int i = 0; i < this.exampleCount; i++)
       {
           final DataType exampleI = this.examples.get(i);

            for (int j = 0; j < this.exampleCount; j++)          {
               
                            final DataType exampleJ = this.examples.get(j);
               final double similarity = -Math.sqrt(this.divergence.evaluate(
                    exampleI, exampleJ));
                
                this.similarities[i][j] = (similarity);
                 
            //  System.out.println("check " +" "+i+" "+j+" "+ similarities[i][j]);
            
           }
       }
        double[] dataValuesnew = new double[(this.exampleCount * this.exampleCount)];
        int k = 0;
        for(int i = 0;i<this.exampleCount;i++)
        {
            for(int j = 0;j<this.exampleCount;j++)
            {
                dataValuesnew[k++] = this.similarities[i][j];
            }
        }
         Median m = new Median();
        Mean me = new Mean();
        Max ma = new Max();
        Min mi = new Min();
        double median = m.evaluate(dataValuesnew);
        double mean = me.evaluate(dataValuesnew);
        double max =  ma.evaluate(dataValuesnew);
        double min =  mi.evaluate(dataValuesnew);
        System.out.println("median " + median);
        System.out.println("mean " + mean);
        System.out.println("max " + max);
        System.out.println("min " + min);
        
         for(int i = 0;i<k;i++)
        {
            
                if(dataValuesnew[i] > max2 && dataValuesnew[i] < max)
                    max2 = dataValuesnew[i];
            
        }
         System.out.println("max2 "+max2);
         
//         double[] dataValuesnew1 = new double[(this.exampleCount * this.exampleCount)];
//         int p = 0;
//         for(int i = 0;i<this.exampleCount;i++)
//        {
//            for(int j = 0;j<this.exampleCount;j++)
//            {
//                if(i == j)
//                    continue;
//                dataValuesnew1[p++] = this.similarities[i][j];
//            }
//        }
//         double max3 =  ma.evaluate(dataValuesnew1);
//         System.out.println("max3 "+max3);
//        double max_pre = Double.NEGATIVE_INFINITY;
//        for(int i = 0;i<this.exampleCount;i++)
//        {
//            for(int j = 0;j<i;j++)
//            {
//                sum =+ -similarities[i][j];
//                if(-similarities[i][j] > max_pre)
//                    max_pre = -similarities[i][j];
//            }
//        }
//      //  sum =+ similarities[0][1] + similarities[0][2];
//        mean = sum/(this.exampleCount);
//        System.out.println("mean "+ mean );
        // Set the self similarity based on the self divergence.
        for (int i = 0; i < this.exampleCount; i++)
        {
            this.similarities[i][i] = -this.selfDivergence;
           //this.similarities[i][i] = max_pre;
           // this.similarities[i][i] = -mean;
            
           // System.out.println("self " + similarities[i][i]);
        }
        //changing preference values based on pmteric ranking
        Scanner ranksFile = null;
        try {
            ranksFile = new Scanner(new File("F:/shivani/AffinityPropagation/pmetricfile.txt"));
        } catch (FileNotFoundException ex) {
            Logger.getLogger(AffinityPropagation.class.getName()).log(Level.SEVERE, null, ex);
        }

       // ArrayList<Double> centroids = new ArrayList<Double>();
       
        while(ranksFile.hasNextLine()){
            String line = ranksFile.nextLine();
            Scanner scanner = new Scanner(line);
            scanner.useDelimiter(" ");
            int i = 0;
            while(scanner.hasNextDouble()){
                int value = scanner.nextInt();
     this.similarities[value][value] = 0;             //class aware preference
                //centroids.add(scanner.nextDouble());
            }
            scanner.close();
        }

        ranksFile.close();
        
        
        this.setAssignments(new int[this.exampleCount]);
        this.setChangedCount(this.exampleCount);
        this.setClusters(new HashMap<>());
        for (int i = 0; i < this.exampleCount; i++)
        {
            this.assignments[i] = -1;
        }

        
        return true;
    }

    @Override
    protected boolean step()
    {
        
        this.updateResponsibilities();
        this.updateAvailabilities();
        this.setChangedCount(0);
        this.updateAssignments();
        return this.getChangedCount() > 0;
    }
    protected void updateResponsibilities()
    {
        
        
        Scanner ranksFile1 = null;
        Scanner ranksFile2 = null;
        Scanner ranksFile3 = null;
        Scanner ranksFile4 = null;
        Scanner ranksFile5 = null;
        Scanner ranksFile6 = null;
         Scanner ranksFile7 = null;
        Scanner ranksFile8 = null;
        try {
            ranksFile1 = new Scanner(new File("F:/shivani/AffinityPropagation/highfile1.txt"));
            ranksFile2 = new Scanner(new File("F:/shivani/AffinityPropagation/highfile2.txt"));
     //       ranksFile3 = new Scanner(new File("F:/shivani/AffinityPropagation/highfile3.txt"));
            ranksFile4 = new Scanner(new File("F:/shivani/AffinityPropagation/lowfile1.txt"));
            ranksFile5 = new Scanner(new File("F:/shivani/AffinityPropagation/lowfile2.txt"));
//            ranksFile6 = new Scanner(new File("F:/shivani/AffinityPropagation/lowfile3.txt"));
//            ranksFile7 = new Scanner(new File("F:/shivani/AffinityPropagation/highfile4.txt"));
//            ranksFile8 = new Scanner(new File("F:/shivani/AffinityPropagation/lowfile4.txt"));
        } catch (FileNotFoundException ex) {
            Logger.getLogger(AffinityPropagation.class.getName()).log(Level.SEVERE, null, ex);
        }
            String l1 = ranksFile1.nextLine();
           // System.out.println(l1);
            String l2 = ranksFile2.nextLine();
          //  System.out.println(l2);
     //       String l3 = ranksFile3.nextLine();
//           System.out.println(l3);
            String l4 = ranksFile4.nextLine();
          //  System.out.println(l4);
            String l5 = ranksFile5.nextLine();
         //   System.out.println(l5);
 //           String l6 = ranksFile6.nextLine();
//            System.out.println(l6);
 //          String l7 = ranksFile7.nextLine();
//            System.out.println(l7);
  //         String l8 = ranksFile8.nextLine();
//           System.out.println(l8);
            String [] str_array1 = l1.split(" ");
            String [] str_array2 = l2.split(" ");
   //        String [] str_array3 = l3.split(" ");
          
            
            String [] str_array4 = l4.split(" ");
            String [] str_array5 = l5.split(" ");
   //        String [] str_array6 = l6.split(" ");
  //          String [] str_array7 = l7.split(" ");
  //          String [] str_array8 = l8.split(" ");
            int[] array1 = new int[str_array1.length+1];
            for(int i=0;i<str_array1.length;i++){
            array1[i] = Integer.parseInt(str_array1[i]);
          //  System.out.println(array1[i]);
            }
               
//            
            int[] array2 = new int[str_array2.length+1];
            for(int i=0;i<str_array2.length;i++){
            array2[i] = Integer.parseInt(str_array2[i]);
           // System.out.println(array2[i]);
            }
//       
//            int[] array3 = new int[str_array3.length+1];
//            for(int i=0;i<str_array3.length;i++){
//            array3[i] = Integer.parseInt(str_array3[i]);
//     //       System.out.println(array3[i]);
//           }
                
            int[] array4 = new int[str_array4.length+1];
            for(int i=0;i<str_array4.length;i++){
            array4[i] = Integer.parseInt(str_array4[i]);
          //  System.out.println(array4[i]);
            }
                
            int[] array5 = new int[str_array5.length+1];
            for(int i=0;i<str_array5.length;i++){
            array5[i] = Integer.parseInt(str_array5[i]);
           // System.out.println(array5[i]);
            }
//                
////                
//            int[] array6 = new int[str_array6.length+1];
//            for(int i=0;i<str_array6.length;i++){
//            array6[i] = Integer.parseInt(str_array6[i]);
////           // System.out.println(array6[i]);
//            }
////        
////            
//            int[] array7 = new int[str_array7.length+1];
//            for(int i=0;i<str_array7.length;i++){
//            array7[i] = Integer.parseInt(str_array7[i]);
////           // System.out.println(array7[i]);
//            }
////            
//            int[] array8 = new int[str_array8.length+1];
//            for(int i=0;i<str_array8.length;i++){
//            array8[i] = Integer.parseInt(str_array8[i]);
////           // System.out.println(array8[i]);
//            }
        double max1 = -1000000000;
             for (int i = 0; i < this.exampleCount; i++)
             {
            for (int k = 0; k < this.exampleCount; k++)
            {

                double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < this.exampleCount; c++)
                {
                    if (c == k)
                    {
                        continue;
                    }

                    final double value =
                       this.availabilities[i][c] + this.similarities[i][c];
                    final double value1 = this.similarities[i][c];

                    if (value > max)
                    {
                        max = value;
                    }
//                    if(value1 > max1)
//                    {
//                        max1 = value1;
//                    }
                }
                
                int class_i = 0;
                int class_k = 0;
                boolean flag = false;
                boolean flagi = false;
                boolean flagk = false;
                // for 2 class data sets
               for(int m = 0;m <str_array1.length;m++)
               {
                   if(array1[m]==k) 
                   {
                       flagk = true;
                   }
                  if(array1[m]== i)
                      flagi = true;
               }
              // i low and k high
//               if(flagi == false && flagk == true)
//               this.similarities[i][k]= max2;
             //  System.out.println("max "+max +" max1 "+ max1);
                // i high/low and k high
               if( flagk == true)                    //hereeerr
               this.similarities[i][k]= max2; 
               
               //i and k low or i low and k high - not good
//               if((flagi == false && flagk == false )|| (flagi == false && flagk == true) )
//               this.similarities[i][k]= -1;
               
               // for multi class dataset
//               for(int n = 0;n < str_array1.length;n++)
//               {
//                   if(array1[n]== i || array2[n]== i || array3[n]== i)//|| array7[n]== i)
//                   {
//                       class_i++;
//                   }
//                   if(array1[n]== k || array2[n]== k || array3[n]== k )//|| array7[n]== k)
//                   {
//                       class_k++;
//                   }
//               }
//              // System.out.println("high class i and high class k "+ class_i +" "+ class_k);
//               if((class_k >= class_i) )//&& (class_k > 1 ))//&& class_i > 1))//&& (class_i != 0 &&(class_k != 0 && class_k != 1)) && (class_i != 1 && (class_k != 1 && class_k != 2)) )
//                   
//               {
//                   flag = true;
//               }
////               
//               if(flag)
//                   this.similarities[i][k] = max2;
                final double responsibility = this.similarities[i][k] - max;
                final double oldResponsibility = this.responsibilities[i][k];
                this.responsibilities[i][k] =
                      this.dampingFactor * oldResponsibility 
                    + this.oneMinusDampingFactor * responsibility;
               // System.out.println("responsibility " + i+ " " + k+ " " + responsibilities[i][k]);
            }
        }
    }

   
    protected void updateAvailabilities()
    {
        for (int i = 0; i < this.exampleCount; i++)
        {
            for (int k = 0; k < this.exampleCount; k++)
            {
                double availability = 0.0;

                for (int j = 0; j < this.exampleCount; j++)
                {
                    if (j == i || j == k)
                    {
                        continue;
                    }

                    final double responsibility = this.responsibilities[j][k];

                    if (responsibility > 0.0)
                    {
                        availability += responsibility;
                    }
                }

                if (i != k)
                {
                    availability += this.responsibilities[k][k];
                    availability = Math.min(0.0, availability);
                }
                final double oldAvailability = this.availabilities[i][k];
                this.availabilities[i][k] =
                      this.dampingFactor * oldAvailability 
                    + this.oneMinusDampingFactor * availability;
              //  System.out.println("availability " + i+ " " + k+ " " + availabilities[i][k]);
            }
        }
    }
    protected void updateAssignments()
    {
        this.setClusters(new HashMap<>());
        for (int i = 0; i < this.exampleCount; i++)
        {
            // Assign the example to the cluster that maximizes a(i,k) + r(i,k).
            int assignment = -1;
            double maximum = Double.NEGATIVE_INFINITY;

            for (int k = 0; k < this.exampleCount; k++)
            {
                final double value =
                    this.availabilities[i][k] + this.responsibilities[i][k];

                if (assignment < 0 || value > maximum)
                {
                   
                    assignment = k;
                    maximum = value;
                }
            }

           
            this.assignCluster(i, assignment);
        }
    }

    protected void assignCluster(
        final int i,
        final int newAssignment)
    {
        // First determine if the assignment has changed.
        final double oldAssignment = this.assignments[i];

        if (newAssignment != oldAssignment)
        {
            // This cluster assignment has changed.
            this.changedCount++;
        }

        this.assignments[i] = newAssignment;

        // Update the cluster memberships.
        final DataType example = this.examples.get(i);

        // Add the example to the new cluster.
        CentroidCluster<DataType> newCluster = this.clusters.get(newAssignment);
        if (newCluster == null)
        {
            // The new cluster does not yet exist so create it.
            final DataType exemplar = this.examples.get(newAssignment);
            newCluster = new CentroidCluster<>(exemplar);
            newCluster.setIndex(newAssignment);
            this.clusters.put(newAssignment, newCluster);
        }

        // Add the example to the new cluster.
        newCluster.getMembers().add(example);
        int[] assignments1 = this.getAssignments();
        
//        k++;
//        for(int y = 0;y<32;y++)
//        {
//        System.out.println( k +" assign"+ assignments1[y]);
//        }
    }

    @Override
    protected void cleanupAlgorithm()
    {
        this.setExamples(null);
        this.setSimilarities(null);
        this.setResponsibilities(null);
        this.setAvailabilities(null);
    }

    @Override
    public ArrayList<CentroidCluster<DataType>> getResult()
    {
        if (this.getClusters() == null)
        {
            return null;
        }
        else
        {
            return new ArrayList<>(
                this.getClusters().values());
        }
    }
    public DivergenceFunction<? super DataType, ? super DataType> 
        getDivergence()
    {
        return this.divergence;
    }
    public void setDivergence(
        final DivergenceFunction<? super DataType, ? super DataType> divergence)
    {
        this.divergence = divergence;
    }
    public double getSelfDivergence()
    {
        return this.selfDivergence;
    }
    public void setSelfDivergence(
        final double selfDivergence)
    {
        this.selfDivergence = selfDivergence;
    }

    public double getDampingFactor()
    {
        return this.dampingFactor;
    }
    public void setDampingFactor(
        final double dampingFactor)
    {
        if (dampingFactor < 0.0 || dampingFactor > 1.0)
        {
            throw new IllegalArgumentException(
                "The damping factor must be between 0.0 and 1.0.");
        }

        this.dampingFactor = dampingFactor;
        this.oneMinusDampingFactor = 1.0 - this.dampingFactor;
    }
    protected ArrayList<DataType> getExamples()
    {
        return this.examples;
    }
    protected void setExamples(
        final ArrayList<DataType> examples)
    {
        this.examples = examples;
        this.exampleCount = examples == null ? 0 : examples.size();
    }
    protected double[][] getSimilarities()
    {
        return this.similarities;
    }
    protected void setSimilarities(
        final double[][] similarities)
    {
        this.similarities = similarities;
    }
    protected double[][] getResponsibilities()
    {
        return this.responsibilities;
    }
    protected void setResponsibilities(
        final double[][] responsibilities)
    {
        this.responsibilities = responsibilities;
    }
    protected double[][] getAvailabilities()
    {
        return this.availabilities;
    }
    protected void setAvailabilities(
        final double[][] availabilities)
    {
        this.availabilities = availabilities;
    }
    protected int[] getAssignments()
    {
        return this.assignments;
    }
    protected void setAssignments(
        final int[] assignments)
    {
        this.assignments = assignments;
    }
    public int getChangedCount()
    {
        return this.changedCount;
    }
    protected void setChangedCount(
        final int changedCount)
    {
        this.changedCount = changedCount;
    }
    protected HashMap<Integer, CentroidCluster<DataType>> getClusters()
    {
        return this.clusters;
    }
    protected void setClusters(
        final HashMap<Integer, CentroidCluster<DataType>> clusters)
    {
        this.clusters = clusters;
    }

    @Override
    public DivergenceFunction<? super DataType, ? super DataType> getDivergenceFunction()
    {
        return this.getDivergence();
    }
    @Override
    public NamedValue<Integer> getPerformance()
    {
        return new DefaultNamedValue<>("number changed", this.getChangedCount());
    }
    
     public double distance(List<Double> instance1, List<Double> instance2) {
        double dist = 0.0;
        k++;
        for (int i = 0; i < instance1.size(); i++) 
            {
            double x = instance1.get(i);
        double y = instance2.get(i);
      //  System.out.println("k "+k);
     //   System.out.println("x " +x);
     //   System.out.println("y "+y);
        if (Double.isNaN(x) || Double.isNaN(y))
        {
            continue; // Mark missing attributes ('?') as NaN.
        }

        dist += (x-y)*(x-y);
        //System.out.println("dist "+dist);
    }

    return Math.sqrt(dist);
}
    public  void calculate() throws Exception
    {
            BufferedReader breader = null;
//       
            breader = new BufferedReader(new FileReader("F:/shivani/dataset/arff/toy.arff"));
            Instances dat = new Instances(breader);
//            EuclideanDistance ed = new EuclideanDistance();
//            
            int countfeature = dat.numAttributes();
             this.setSimilarities(new double[countfeature][countfeature]);
           for(int numAtt = 0; numAtt < countfeature;numAtt++)
           {
               
                 for(int numAtt1 = 0; numAtt1 < countfeature;numAtt1++)
                 {
                     System.out.println("here");
                      List<Double> list1 = new ArrayList<>();
                    List<Double> list2 = new ArrayList<>();
            
                    for(int numSpl = 0;numSpl < dat.numInstances();numSpl++)
                    { 
                        list1.add(dat.instance(numSpl).value(numAtt));
                        list2.add(dat.instance(numSpl).value(numAtt1));
                        
                    }
                    double d  = distance(list1,list2);
                 //   System.out.println("distance "+d);
                   this.similarities[numAtt][numAtt1] = -(d);
                   // mat[numAtt][numAtt1] = d;
                    //    for(int i = 0;i<10;i++)
                      //  System.out.println(d + "distance ");
                
                
                }
                
           }
         
           for(int y = 0;y<countfeature;y++)
           {
               for(int yy = 0;yy <countfeature;yy++)
               {
                   System.out.println("yoohoo "+y*yy+" "+this.similarities[y][yy]);
               }
           }
           // double d = distance(list1,list2);
          //  System.out.println("euclidean distance "+ d);
    }
    
    public static void main(String args[]) throws Exception
    {
        
         
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("F:/shivani/dataset/arff/chen-2002.arff"));
        Instances d = new Instances(breader);
        d.setClassIndex(d.numAttributes()-1);
        
        
        myVector[] dv= new myVector[d.numAttributes()];
        for(int numAtt = 0;numAtt< d.numAttributes();numAtt++)
            {
                double[] values1 = new double[d.numInstances()];
                values1 = d.attributeToDoubleArray(numAtt);
                //System.out.println(dataMedian[p++]);
                dv[numAtt]=new myVector(values1);
            
            }
        
//        Vector3[] data = new Vector3[]{
//        
//               new Vector3(0.0,1.0,2.0),
//               new Vector3(0.0,2.0,4.0),            // square clustering
//               new Vector3(0.0,4.0,8.0),
//               new Vector3(0.0,5.0,10.0),
//               new Vector3(0.0,6.0,12.0)
//               
//               
//            };
//        double[] dataValues = new double[(d.numAttributes() -1) * d.numInstances()];
//        int k = 0;
//        for(int i = 0;i<d.numAttributes() -1;i++)
//        {
//            for(int j = 0;j<d.numInstances();j++)
//            {
//                dataValues[k] = d.instance(j).value(i);
//                //dataValues[k] =this.similarities[i][j];
//                //System.out.println(dataValues[k]);
//                k++;
//            }
//        }
        
//        Median m = new Median();
//        Mean me = new Mean();
//        Max ma = new Max();
//        Min mi = new Min();
//        double median = m.evaluate(dataValues);
//        double mean = me.evaluate(dataValues);
//        double max =  ma.evaluate(dataValues);
//        double min =  mi.evaluate(dataValues);
//        System.out.println("median " + median);
//        System.out.println("mean " + mean);
//        System.out.println("max " + max);
//        System.out.println("min " + min);
        System.out.println("Attributes "+ d.numAttributes() + " Instances "+ d.numInstances() +" Classes " + d.numClasses());
       AffinityPropagation< Vectorizable> instance = new AffinityPropagation<>(
             EuclideanDistanceSquaredMetric.INSTANCE,15);
    
  // Collection<CentroidCluster<Vectorizable>> clusters = instance.learn(Arrays.asList(data));
        Collection<CentroidCluster<Vectorizable>> clusters = instance.learn(Arrays.asList(dv));
        System.out.println("No of clusters "+ clusters.size());
       // instance.calculate();
        
        File file = new File("F:/shivani/AffinityPropagation/clustersFile.txt");
 File file1 = new File("F:/shivani/AffinityPropagation/clustersFile1.txt");
 
			// if file doesnt exists, then create it
			if (!file.exists()) {
				file.createNewFile();
			}
 
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
                        if (!file1.exists()) {
				file1.createNewFile();
			}
 
			FileWriter fw1 = new FileWriter(file1.getAbsoluteFile());
			BufferedWriter bw1 = new BufferedWriter(fw1);
        clusters.stream().forEach((cluster) -> {
        System.out.println(d.attribute(cluster.getIndex()).name());
     //  System.out.println(cluster.getIndex());
     // System.out.println("Index :"+cluster.getIndex()+" Attribute name :"+d.attribute(cluster.getIndex()).name().replaceAll("at(.*)", "at")+" Centroid values :"+cluster.getCentroid() + "...");
         try {
             String content =Integer.toString(cluster.getIndex());
             bw.write(content + " ");
		
              } catch (IOException e) {
		}
            });d.attribute(DEFAULT_ITERATION).name();
        System.out.println("Done");
        bw.close();
    
        Scanner clustersFile = new Scanner(new File("F:/shivani/AffinityPropagation/clustersFile.txt"));
        
        
         
        
        int size = clusters.size();
        int[] remArray = new int[size +1];
        //int[] remArray = new int[size];
        while(clustersFile.hasNextLine()){
        String line = clustersFile.nextLine();
        Scanner scanner = new Scanner(line);
        scanner.useDelimiter(" ");
        int i = 0;
        while(scanner.hasNextDouble()){
            int n = scanner.nextInt();
            if(n == d.numAttributes()-1)
            {
                remArray[i++]=  d.numAttributes() - 2;
            }
            else
            remArray[i++] = n;
                //centroids.add(scanner.nextDouble());
            }
      //  remArray[7] = 1286;
       
        scanner.close();
        }

        clustersFile.close();
      
        Instances       inst;
        Instances       instNew;
        Remove          remove;
     
        inst   = new Instances(new BufferedReader(new FileReader("F:/shivani/dataset/arff/chen-2002.arff")));
       // RenameAttribute ra = new RenameAttribute();
        //ra.
        remArray[clusters.size()] = inst.numAttributes()-1;
        // remArray[11] = 83;
      //  remArray[106] = 1736; //i +" "+ remArray[i] +" "+ 
        for(int i = 0;i<clusters.size();i++)
        System.out.println(d.attribute(remArray[i]).name());
        remove = new Remove();
        remove.setAttributeIndicesArray(remArray);
        remove.setInvertSelection(true);
        remove.setInputFormat(inst);
        instNew = Filter.useFilter(inst, remove);
        instNew.setClassIndex(instNew.numAttributes()-1);
        
        inst.setClassIndex(inst.numAttributes()-1);
        Classifier cls = new J48();
        SMO sm = new SMO();
        NaiveBayes nb = new NaiveBayes();
        Evaluation eval = new Evaluation(instNew);
        Evaluation eval1 = new Evaluation(instNew);
        Evaluation eval2 = new Evaluation(instNew);
        // Evaluation eval = new Evaluation(inst);
        Debug.Random rand = new Debug.Random(1);  // using seed = 1
        int folds = 10;
         eval.crossValidateModel(sm, instNew, folds, rand);
         
      //  eval1.crossValidateModel(nb, instNew, folds, rand);
      //  eval2.crossValidateModel(cls, instNew, folds, rand);
       // eval.crossValidateModel(cls, inst, folds, rand);
         
         int o = (int)eval.correct();
         String c =Integer.toString(o);
             bw1.write(c);
		bw1.close();
      //   System.out.println("here" +eval.correct());
        // bw1.write(o);
        System.out.println(eval.toSummaryString());
        
//        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
//
//      //read a line from the console
//      String lineFromInput = in.readLine();
//
//    PrintStream out = new PrintStream(new FileOutputStream("clutersFile1.txt"));
//        System.setOut(out);
//      out.close();
  
        
        
       // bw1.write(eval.toSummaryString());
     //   System.out.println(eval1.toSummaryString());
     //   System.out.println(eval2.toSummaryString());

    }
}
            