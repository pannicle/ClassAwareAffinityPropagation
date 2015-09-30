package featurerank;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.Capabilities;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
//import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.trees.RandomTree;
import java.util.Enumeration;
import java.util.Vector;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.ASEvaluation;
public class InfoGainAttributeEval extends ASEvaluation
  implements AttributeEvaluator{                                                 

 
  private boolean m_missing_merge;
 
  
  private boolean m_Binarize;
 
 
  private double[] m_InfoGains;
 

  public InfoGainAttributeEval () {
    resetOptions();
  }
 
  
  public void setBinarizeNumericAttributes (boolean b) {
    m_Binarize = b;
  }
   /**
   * Returns the capabilities of this evaluator.
   
   */
  public Capabilities getCapabilities() {
    Capabilities result;
        result = super.getCapabilities();
    result.disableAll();
    
    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.NOMINAL_CLASS);
    return result;
  }
  public void buildEvaluator (Instances data)
    throws Exception {
  getCapabilities().testWithFail(data);
 
   int classIndex = data.classIndex();
    //data.setClassIndex(0);
   System.out.println("the index is    " );
   System.out.println(classIndex);
   System.out.println("Attributes " +data.numAttributes());
    int numInstances = data.numInstances();
    System.out.println("inst" + numInstances);
   if (!m_Binarize) 
    {
      Discretize disTransform = new Discretize();
      disTransform.setUseBetterEncoding(true);
      disTransform.setInputFormat(data);
      data = Filter.useFilter(data, disTransform);
    } 
    else 
    {
      NumericToBinary binTransform = new NumericToBinary();
      binTransform.setInputFormat(data);
      data = Filter.useFilter(data, binTransform);
    }      
    int numClasses = data.attribute(classIndex).numValues();
 
    // Reserve space and initialize counters
    double[][][] counts = new double[data.numAttributes()][][];
    for (int k = 0; k < data.numAttributes(); k++) {
      if (k != classIndex) {
        int numValues = data.attribute(k).numValues();
        System.out.println("attr values " +data.attribute(k));
        
        System.out.println("num values " + numValues);
        
        counts[k] = new double[numValues + 1][numClasses + 1];
      }
    }
 System.out.println("num classes" + numClasses);
    // Initialize counters
    double[] temp = new double[numClasses + 1];
    for (int k = 0; k < numInstances; k++) {
      Instance inst = data.instance(k);
     // System.out.println("instance " + data.instance(4));
      if (inst.classIsMissing()) {
        temp[numClasses] += inst.weight();
      } else {
        temp[(int)inst.classValue()] += inst.weight();
      }
    }
    for (int k = 0; k < counts.length; k++) {
      if (k != classIndex) {
        for (int i = 0; i < temp.length; i++) {
          counts[k][0][i] = temp[i];
        }
      }
    }
 
    // Get counts
    for (int k = 0; k < numInstances; k++) {
      Instance inst = data.instance(k);
      for (int i = 0; i < inst.numValues(); i++) {
        if (inst.index(i) != classIndex) {
          if (inst.isMissingSparse(i) || inst.classIsMissing()) {
            if (!inst.isMissingSparse(i)) {
              counts[inst.index(i)][(int)inst.valueSparse(i)][numClasses] += 
                inst.weight();
              counts[inst.index(i)][0][numClasses] -= inst.weight();
            } else if (!inst.classIsMissing()) {
              counts[inst.index(i)][data.attribute(inst.index(i)).numValues()]
                [(int)inst.classValue()] += inst.weight();
              counts[inst.index(i)][0][(int)inst.classValue()] -= 
                inst.weight();
            } else {
              counts[inst.index(i)][data.attribute(inst.index(i)).numValues()]
                [numClasses] += inst.weight();
              counts[inst.index(i)][0][numClasses] -= inst.weight();
            }
          } else {
            counts[inst.index(i)][(int)inst.valueSparse(i)]
              [(int)inst.classValue()] += inst.weight();
            counts[inst.index(i)][0][(int)inst.classValue()] -= inst.weight();
          }
        }
      }
    }
   
    m_InfoGains = new double[data.numAttributes()];
    for (int i = 0; i < data.numAttributes(); i++) {
      if (i != classIndex) {
        m_InfoGains[i] = 
          (ContingencyTables.entropyOverColumns(counts[i]) 
           - ContingencyTables.entropyConditionedOnRows(counts[i]));
      }
    }
  }
  protected void resetOptions () {
    m_InfoGains = null;
    m_missing_merge = true;
    m_Binarize = false;
  }
  public double evaluateAttribute (int attribute)
    throws Exception {
 
    return m_InfoGains[attribute];
  }
 
  void info (String[] args) throws Exception 
  {
         BufferedReader breader = null;
         breader = new BufferedReader(new FileReader("H:/chowdary-2006_database1.arff"));
         Instances data = new Instances(breader);
         runEvaluator(new InfoGainAttributeEval(), args);
         //data.setClassIndex(0);
          AttributeSelection attr = new AttributeSelection();
          InfoGainAttributeEval eval1 = new InfoGainAttributeEval();
          Ranker search1 = new Ranker();  
          attr.setEvaluator(eval1);
          attr.setSearch(search1);
          attr.SelectAttributes(data);
          System.out.println(attr.toResultsString());
          
  
  }
}