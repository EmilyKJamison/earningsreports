/*******************************************************************************
 * Copyright 2015
 * Ubiquitous Knowledge Processing (UKP) Lab
 * Technische Universität Darmstadt
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/
package de.tudarmstadt.ukp.experiments.ej.ieearningsreports.FeatureExtractors;

import static org.apache.uima.fit.util.JCasUtil.selectCovered;

import java.util.HashSet;
import java.util.Set;

import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.ADJ;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.ADV;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.ART;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.CARD;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.CONJ;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.N;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.O;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.PP;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.PR;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.PUNC;
import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.V;
import de.tudarmstadt.ukp.dkpro.tc.api.exception.TextClassificationException;
import de.tudarmstadt.ukp.dkpro.tc.api.features.ClassificationUnitFeatureExtractor;
import de.tudarmstadt.ukp.dkpro.tc.api.features.Feature;
import de.tudarmstadt.ukp.dkpro.tc.api.features.FeatureExtractorResource_ImplBase;
import de.tudarmstadt.ukp.dkpro.tc.api.type.TextClassificationUnit;

/**
 * Extracts the ratio of each universal POS tags to the total number of tags 
 */
public class POSRatioFeatureExtractor
    extends FeatureExtractorResource_ImplBase
    implements ClassificationUnitFeatureExtractor
{
    public static final String FN_ADJ_RATIO = "AdjRatioFeature";
    public static final String FN_ADV_RATIO = "AdvRatioFeature";
    public static final String FN_ART_RATIO = "ArtRatioFeature";
    public static final String FN_CARD_RATIO = "CardRatioFeature";
    public static final String FN_CONJ_RATIO = "ConjRatioFeature";
    public static final String FN_N_RATIO = "NRatioFeature";
    public static final String FN_O_RATIO = "ORatioFeature";
    public static final String FN_PP_RATIO = "PpRatioFeature";
    public static final String FN_PR_RATIO = "PrRatioFeature";
    public static final String FN_PUNC_RATIO = "PuncRatioFeature";
    public static final String FN_V_RATIO = "VRatioFeature";

    @Override
    public Set<Feature> extract(JCas jcas, TextClassificationUnit classificationUnit)
        throws TextClassificationException
    {
        Set<Feature> features = new HashSet<Feature>();

        double total = JCasUtil.selectCovered(jcas, POS.class, classificationUnit).size();
        double adj = selectCovered(jcas, ADJ.class, classificationUnit).size() / total;
        double adv = selectCovered(jcas, ADV.class, classificationUnit).size() / total;
        double art = selectCovered(jcas, ART.class, classificationUnit).size() / total;
        double card = selectCovered(jcas, CARD.class, classificationUnit).size() / total;
        double conj = selectCovered(jcas, CONJ.class, classificationUnit).size() / total;
        double noun = selectCovered(jcas, N.class, classificationUnit).size() / total;
        double other = selectCovered(jcas, O.class, classificationUnit).size() / total;
        double prep = selectCovered(jcas, PP.class, classificationUnit).size() / total;
        double pron = selectCovered(jcas, PR.class, classificationUnit).size() / total;
        double punc = selectCovered(jcas, PUNC.class, classificationUnit).size() / total;
        double verb = selectCovered(jcas, V.class, classificationUnit).size() / total;

        features.add(new Feature(FN_ADJ_RATIO, adj));
        features.add(new Feature(FN_ADV_RATIO, adv));
        features.add(new Feature(FN_ART_RATIO, art));
        features.add(new Feature(FN_CARD_RATIO, card));
        features.add(new Feature(FN_CONJ_RATIO, conj));
        features.add(new Feature(FN_N_RATIO, noun));
        features.add(new Feature(FN_O_RATIO, other));
        features.add(new Feature(FN_PR_RATIO, pron));
        features.add(new Feature(FN_PP_RATIO, prep));
        features.add(new Feature(FN_PUNC_RATIO, punc));
        features.add(new Feature(FN_V_RATIO, verb));

        return features;
    }
}