/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    RSARSubsetEval.java
 *    Copyright (C) 2011 University of Aberystwyth, Dyfed, Wales, UK.
 *
 */

package  weka.attributeSelection;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;

/** 
 <!-- globalinfo-start -->
 * RSARSubsetEval :<br/>
 * An implementation of the QuickReduct algorithm of rough set attribute reduction (RSAR).<br/>
 * Applicable for use on large gene expression (GE) datasets with numeric continuous data.<br/>
 * Evaluates subsets using rough set dependency, to return a feature subset giving only the rough set positive region.<br/>
 * Feature subset evaluation merit value between 0.0 and 1.0. Not all datasets will reach maximum dependency of 1.0.<br/>
 * 
 * Algorithm:
 * On building, discretizes data using Fayyad's MDL, with betterEncoding enabled.
 * 
 * 1) Hashes each feature subset, storing: 
 * 		- hashed approximation of each unique "instance signature", with: 
 * 		- decision value
 * 		- number of matching instance signatures
 * 		- flag to denote an inconsistency in instance signature decision values. 
 * 2) Evaluates hash to determine dependency:
 * 		- Count all consistent instance signatures.
 * 		- dependency is given by 'count' / 'total_num_instances' (in the range 0.0-1.0)
 * 3) Return dependency value
 * 
 * 
 * Time complexity: 	O(n*(c+c)*P(m)) 
 * - n=instances; (c+c)=double hashing time cost; P(m) power-set of conditional attributes.
 * 
 * Space complexity for each subset evaluation:		O((u(n)*~1.33*c)+h)   	
 * - u(n)= unique instance signatures; ~1.33 = hashtable size increase; c = double[3] capacity; j= java overhead for hashtable, double array, etc.
 * 
 * For more information see:<br/>
 * <br/>
 * A. Chouchoulas, and Q. Shen, “Rough set-aided keyword reduction for text categorization,” Applied Artificial Intelligence: An International Journal, vol. 15, no. 9, pp. 843-873, 2001.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * ProCite:
 * <pre>
 	TY - JOUR
	JO - Applied Artificial Intelligence: An International Journal
	PB - Taylor & Francis
	AU - Chouchoulas, Alexios
	AU - Shen, Qiang
	TI - Rough set-aided keyword reduction for text categorization
	SN - 0883-9514
	PY - 2001
	VL - 15
	IS - 9
	SP - 843
	EP - 873
	UR - http://www.informaworld.com/10.1080/088395101753210773
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 *
 * @author Peter Scully  (pds7 at aber dot ac dot uk)
 * @version $Revision: 1.0 $
 * 	v3/1.0: 	- adds removes debug info.
 * 	v2/0.2:		- adds minimised hash / Double array design, with debugging info.
 * 	v1/0.1: 	- initial OO design
 * 			
 * @see Discretize
 */
public class RSARSubsetEval
extends ASEvaluation
implements SubsetEvaluator, 
TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = 747878400813276317L;

	/** The training instances */
	private Instances m_trainInstances;
	/** Discretise attributes when class in nominal */
	private Discretize m_disTransform;
	/** The class index */
	private int m_classIndex;



	//=============================================================================
	// Variables:
	//=============================================================================
	private Hashtable<Integer, double[]> hash;

	/** Index for an count of tuple occurrences, used in the double[3] value of the hashmap. */
	private static final int INDEX_COUNT 			= 2;
	/** Index for an decision value of first tuple's occurrence, used in the double[3] value of the hashmap. */
	private static final int INDEX_DECISION_VALUE 	= 0;
	/** Index for an inconsistency flag, used in the double[3] value of the hashmap. */
	private static final int INDEX_CONSISTENCY 		= 1;
	/** private Enum representing all possible consistency values, used in the double[3] value of the hashmap, at position index INDEX_CONSISTENCY. */
	private enum ConsistencyFlag{ Consistent, Inconsistent }
	/** Debug */
	private boolean show_debug_content = false;






	/**
	 * Returns a string describing this attribute evaluator
	 * @return a description of the evaluator suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "RSARSubsetEval :\n"
		+"An implementation of the QuickReduct algorithm of rough set attribute reduction (RSAR).\n"
		+"Applicable for use on large gene expression (GE) datasets with numeric continuous data.\n"
		+"Evaluates subsets using rough set dependency, to return a feature subset giving only the rough set positive region.\n"
		+"Feature subset evaluation merit value between 0.0 and 1.0. Not all datasets will reach maximum dependency of 1.0.\n"
		+ "For more information see:\n\n"
		+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation        result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "A. Chouchoulas, and Q. Shen");
		result.setValue(Field.YEAR, "2001");
		result.setValue(Field.TITLE, "Rough set-aided keyword reduction for text categorization");
		result.setValue(Field.JOURNAL, "Applied Artificial Intelligence: An International Journal");
		result.setValue(Field.VOLUME, "15");
		result.setValue(Field.NUMBER, "9");
		result.setValue(Field.PAGES, "843-873");
		return result;
	}

	/**
	 * Constructor
	 */
	public RSARSubsetEval () {
		resetOptions();
	}


	/**
	 * Returns the capabilities of this evaluator.
	 *
	 * @return            the capabilities of this evaluator
	 * @see               Capabilities
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.MISSING_VALUES);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);

		return result;
	}












	//=============================================================================
	// buildEvaluator()
	//=============================================================================

	/**
	 * Generates a attribute evaluator. Has to initialize all fields of the 
	 * evaluator that are not being set via options.
	 *
	 * RSAR discretises attributes (always):
	 * - Uses Fayyad &amp; Irani's MDL method (the default)
	 * - Uses betterEncoding.
	 *
	 * @param data set of instances serving as training data 
	 * @throws Exception if the evaluator has not been 
	 * generated successfully
	 */
	public void buildEvaluator (Instances data)
	throws Exception {
		show_debug_content 	= false;
		
		// can evaluator handle data?
		getCapabilities().testWithFail(data);

		m_trainInstances 	= new Instances(data);
		m_trainInstances.deleteWithMissingClass();
		m_classIndex 		= m_trainInstances.classIndex();

		//Discretize:
		m_disTransform 		= new Discretize();
		m_disTransform.setUseBetterEncoding(true);
		m_disTransform.setInputFormat(m_trainInstances);
		m_trainInstances 	= Filter.useFilter(m_trainInstances, m_disTransform);
	}







	//=============================================================================
	// evaluateSubset()
	//=============================================================================

	/**
	 * evaluates a subset of attributes.
	 *
	 * @param subset a bitset representing the attribute subset to be 
	 * evaluated 
	 * @return the merit (in range of 0.0 to 1.0)
	 * @throws Exception if the subset could not be evaluated
	 */
	public double evaluateSubset (BitSet subset) throws Exception {
		double merit 	= 0.0;
		Double[] tupleValues;
		double decisionValue;
		reset();
		
		//----------- DEBUG ONLY:
		StringBuffer str 		= new StringBuffer(); 
		//----------- DEBUG ONLY:

		//If featureSubset is not empty set:
		if(subset.length() > 0){

			Enumeration<Instance> iterator 	= m_trainInstances.enumerateInstances();
			while( iterator.hasMoreElements() ){
				Instance inst 	= iterator.nextElement();
				tupleValues 	= getTupleValues(subset, inst);
				decisionValue 	= inst.classValue();
				addNewTuple(decisionValue, tupleValues);
			}
			merit = getMerit();

			
			if(show_debug_content == true){	//----------- DEBUG ONLY:
				str 	= Debug_getAttributeIndexes(subset);
			}					//----------- DEBUG ONLY:
		}

		
		if(show_debug_content == true){	//----------- DEBUG ONLY:
			System.out.print("Subset: ");
			System.out.print(str.toString());
			System.out.print("\t = "+merit);
			System.out.println();
		}					//----------- DEBUG ONLY:
		
		return  merit;
	}

	//----------------------------------------------------------------------
	//--------- Debug only, can be deleted.
	//----------------------------------------------------------------------
	private StringBuffer Debug_getAttributeIndexes(BitSet subset){
		StringBuffer str 		= new StringBuffer(); 
		for (int indexAttr=subset.nextSetBit(0); indexAttr>=0; indexAttr=subset.nextSetBit(indexAttr+1)) {	//i = attribute index
			str.append(indexAttr+", ");
		}
		return str;
	}
	//----------------------------------------------------------------------
	//----------------------------------------------------------------------


	/** Get the featureSubset tuple values.
	 * @return as array of Doubles.
	 */
	private Double[] getTupleValues(BitSet subset, Instance inst) throws Exception{  
		Double[] tupleValues 	= new Double[subset.length()]; 
		int indexValues			= 0;
		
		for (int indexAttr=subset.nextSetBit(0); indexAttr>=0; indexAttr=subset.nextSetBit(indexAttr+1)) {	//Foreach attributeIndex of the featureSubset
			if (indexAttr == m_classIndex) {
				throw new Exception("Subset should not contain decision attribute!");
			}
			tupleValues[indexValues++] 	= inst.value(indexAttr);
		}

		return tupleValues;
	}

	/** Get the merit value, the "Dependency Degree Value".
	 * 
	 * @return merit
	 */
	private double getMerit(){
		double merit	= 0.0;
		double num 		= (double) countConsistentInstances();//get accumulated number of consistent instances.
		double denom 	= (double) m_trainInstances.numInstances();	//get number of instances.

		if(denom < 0.0) {				//Handle unexpected boundary cases:
			denom *= -1.0;
		}
		if(denom != 0.0) {
			merit = (num/denom);		//Merit of subset:
			if(merit < 0.0) {			//Handle unexpected Merit boundary cases:
				merit *= -1.0;
			}
		}

		if(show_debug_content == true){	//----------- DEBUG ONLY:
			System.out.print("\t (Merit="+num+"/"+denom+")\t");
		}					//----------- DEBUG ONLY:
		return merit;
	}

	/** Count the quantity of consistent instances, return as double.
	 * 
	 * @return
	 */
	private double countConsistentInstances(){
		double count = 0.0;
		Iterator<double[]> iterator = hash.values().iterator();
		while( iterator.hasNext() ){
			double[] thisValue = iterator.next();
			count += getConsistencyCount(thisValue);
		}	
		return count;
	}

	/** Add new tuple and decision value into Hashtable data structure. 
	 * If tuple already exists with same class, it increments the occurence count. 
	 * If tuple already exists with different class, it increments the occurence count and marks as inconsistent.
	 * 
	 * @param decisionValue
	 * @param tuple
	 */
	private void addNewTuple(double decisionValue, Double[] tuple){
		if( checkKeyExists(tuple) ){                                           
			double[] value 	= getHashValue(tuple);
			if( equal_DecisionValues(value, decisionValue)	){  				
				value 		= setInconsistent(value);                             
			}
			value = incrementCount(value);                                     
		}
		else{
			double[] value  = createValueArray(decisionValue);
			Integer key 	= getTupleHashCode(tuple);
			hash.put(key, value);                                              
		}
	}

	/** Create the Hashtable key's value array. Array contains default values for occurrences count, consistency flag, and the @param decisionValue argument.  
	 * 
	 * @param decisionValue
	 * @return
	 */
	private double[] createValueArray(double decisionValue){
		double[] value 				= new double[3];
		value[INDEX_DECISION_VALUE] = decisionValue;                                
		value[INDEX_CONSISTENCY]	= (double) ConsistencyFlag.Consistent.ordinal();                           				
		value[INDEX_COUNT] 			= 1;                   
		return value;
	}
	
	/** Determine if @param is set as consistent.
	 * 
	 * @param value - Hashtable key's "value"
	 * @return
	 */
	private boolean isConsistent(double[] value){
		return ( new Double(value[INDEX_CONSISTENCY]).compareTo( new Double((double) ConsistencyFlag.Consistent.ordinal()) ) == 0 ); 
	}

	/** Flag Hashtable key's value as inconsistent.
	 *  
	 * @param values
	 * @return the altered array.
	 */
	private double[] setInconsistent(double[] values){
		values[INDEX_CONSISTENCY] = (double) ConsistencyFlag.Inconsistent.ordinal();
		return values;
	}
	
	/** Get the count of instances for this @param, if its consistency flag is set.
	 * 
	 * @param value - Hashtable key's "value"
	 * @return
	 */
	private double getConsistencyCount(double[] value){    
		return isConsistent(value) ? value[INDEX_COUNT] : 0.0; 
	}

	/** Increment Hashtable key's value's tuple occurrences counter.
	 * @param values
	 * @return the altered array.
	 */
	private double[] incrementCount(double[] values){
		values[INDEX_COUNT]++;
		return values;
	}
	
	/** Compare if @param value's decision value is equal to the @param classLabel.
	 * 
	 * @param value - Hashtable key's value
	 * @param classLabel - This tuple occurence's decision value
	 * @return True if decision value match, else false.
	 */
	private boolean equal_DecisionValues(double[] value, double classLabel){
		return ( new Double(getDecisionValue(value) ).compareTo( new Double(classLabel) ) == 0);
	}
	
	/** Get decision value stored within Hashtable key's "value". 
	 * @param value
	 * @return
	 */
	private double getDecisionValue(double[] value){
		return value[INDEX_DECISION_VALUE];
	}
	
	/** Checks if key exists.
	 * 
	 * @param c
	 * @return
	 */
	private boolean checkKeyExists(Double[] key){
		return hash.containsKey(getTupleHashCode(key));
	}

	
	/** Generate hash-able tuple key.
	 * @todo consider re-implementing hashtable hashing function to reduce this secondary hashing.
	 * 
	 * @param key
	 * @return hashcode
	 */
	private Integer getTupleHashCode(Double[] key){
		return Arrays.deepHashCode(key);
	}

	
	/** Get Hashtable key's value:
	 * @param tuple
	 * @return
	 */
	private double[] getHashValue(Double[] tuple){
		return hash.get(getTupleHashCode(tuple));
	}
	
	/** Reset the HashTable data structure.
	 * 	- Uses initial size of Num_Instances*(1/LoadFactor) to reduce rehashing possibilities.
	 *  
	 * @todo consider space complexity on datasets with large number of instances. (e.g. 1 million).  
	 */
	private void reset(){
		double load_factor	= 0.75;
		int initial_size 	= (int) Math.ceil(m_trainInstances.size()*(1/load_factor));
		hash 				= new Hashtable<Integer, double[]>(initial_size);
	}


	//=============================================================================
	// postProcess()
	//=============================================================================


	/**
	 * Provides a chance for a attribute evaluator to do any special
	 * post processing of the selected attribute set.
	 *
	 * @param attributeSet the set of attributes found by the search
	 * @return a possibly ranked list of postprocessed attributes
	 * @exception Exception if postprocessing fails for some reason
	 */
	public int [] postProcess(int [] attributeSet) 
	throws Exception {

		System.out.println("PP:Attr(s):"+print_getPostProcess_AttributeIndexes(attributeSet) );
		System.out.println("PP:Length:"+attributeSet.length );
		return attributeSet;
	}

	/** Prepare a StringBuffer of the selected attribute subset.
	 * @param subset
	 * @return
	 */
	private StringBuffer print_getPostProcess_AttributeIndexes(int[] subset){
		StringBuffer str 		= new StringBuffer();
		str.append("[ ");
		for (int i = 0; i < subset.length; i ++) {	//i = attribute index
			str.append(i +", ");
		}
		str.append(" ]");
		return str;
	}













	/**
	 * returns a string describing RSAR
	 *
	 * @return the description as a string
	 */
	public String toString () {
		StringBuffer text = new StringBuffer();

		if (m_trainInstances == null) {
			text.append("RSAR subset evaluator has not been built yet\n");
		}
		else {
			text.append("\tRSAR Subset Evaluator\n");
		}
		return  text.toString();
	}





	protected void resetOptions () {
		m_trainInstances = null;
	}


	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1.0 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param args the options
	 */
	public static void main (String[] args) {
		runEvaluator(new RSARSubsetEval(), args);
	}	
}


