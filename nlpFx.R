library(tm)
library(textstem)
library(openNLP)
library(topicmodels)
library(ggplot2)
library(dplyr)
library(tidyr)


GetDtm = function(x, textMineRVersion = F){
  txt = as.vector(x)
  corp = Corpus(VectorSource(txt))
  corp = tm_map(corp, removeNumbers)
  corp = tm_map(corp, function(y) removeWords(y,stopwords()))
  corp = tm_map(corp, tolower)
  corp = tm_map(corp, removePunctuation)
  corp = tm_map(corp, stripWhitespace)
  corp = tm_map(corp, textstem::lemmatize_strings)
  if (textMineRVersion){
    txtVec = sapply(1:length(corp), function(i) corp[[i]]$content)
    return(textmineR::CreateDtm(doc_vec = txtVec,
                                doc_names = 1:length(txtVec),
                                ngram_window = c(1,1)))
  }
  dtm = DocumentTermMatrix(corp)
  if (any(apply(dtm, 1, sum) == 0)){
    # get documents that do not contain any information after the above removals and set text to "(Blank)" to avoid errors, then rerun prep
    x[apply(dtm, 1, sum) == 0] = "(Blank)"
    return(GetDtm(x))
  }
  return(dtm)
}

dtmAsMatrix = function(dtm) as.matrix(dtm)

GetTfIdf = function(dtm){
  dtm = dtmAsMatrix(dtm)
  tf = t(apply(dtm, 1, function(x) x / sum(x)))
  idf = apply(dtm, 2, function(x) log(length(x) / sum(x > 0)))
  t(t(tf) * idf)
}

innerClusterTfIdf = function(dtm, clust){
  dtm = dtmAsMatrix(dtm)
  # add all documents from a cluster together then perform tfidf
  iDtm = t(sapply(unique(clust), function(x){
    i = which(clust == x)
    apply(dtm[i,,drop=F], 2, sum)
  }))
  GetTfIdf(iDtm)
}

sent_token_annotator = Maxent_Sent_Token_Annotator()
word_token_annotator = Maxent_Word_Token_Annotator()
posAnnotator = Maxent_POS_Tag_Annotator()

getPos = function(s, sentAnnotator, wordAnnotator, posAnnotator){
  s1a = annotate(s, list(sentAnnotator, wordAnnotator))
  s1 = annotate(s, posAnnotator, s1a)
  
  word_subset = subset(s1, type=='word')
  tags = sapply(word_subset$features , '[[', "POS")
  words = sapply(word_subset, function(x) substr(s, x$start, x$end))
  return(data.frame(words, tags))
}

FilterTextByPos = function(x, tags = c("JJ", "NN", "NNS", "NNP", "NNPS")){ # defaulting to filtering on Adjectives and nouns
  txt = as.vector(x)
  sapply(txt, function(y) {
    df = getPos(y, sent_token_annotator, word_token_annotator, posAnnotator)
    return(paste0(df$words[df$tags %in% tags], collapse = " ")) # returns words that match tags separated by a space as a string
  })
}

EvaluateKs = function(textVec, sampleIndices, ks = 2:20){
  
  dtm = GetDtm(textVec)
  dtm_train = dtm[sampleIndices,]
  dtm_test = dtm[-sampleIndices,]
  
  dtmDgc = GetDtm(textVec, T)
  dtmDgc_train = dtmDgc[sampleIndices,]
  dtmDgc_test = dtmDgc[-sampleIndices,]
  
  #dtm_train = GetDtm(trainSet)
  #dtm_test = GetDtm(testSet)
  #dtmDgc_train = GetDtm(trainSet, T)
  #dtmDgc_test = GetDtm(testSet, T)
  
  
  #1
  modelList = tm_parLapply(ks, function(k){
    m = textmineR::FitLdaModel(dtm = dtmDgc_train,
                               k = k,
                               iterations = 220,
                               burnin = 180,
                               #beta = colSums(reviewDtm_train) / sum(reviewDtm_train) * 100,
                               optimize_alpha = T,
                               cpus = 1)
    m$k = k
    return(m)
  })
  
  coherence_matrix = data.frame(k = sapply(modelList, function(x) nrow(x$phi)),
                                coherence = sapply(modelList, function(x) mean(x$coherence)))
  
  g1 = ggplot(coherence_matrix) +
    geom_line(aes(x = k, y = coherence)) +
    theme(axis.text.y = element_blank())
  
  #2
  system.time({
    metricsView = ldatuning::FindTopicsNumber(dtm_train, 
                                              topics = ks, 
                                              metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
                                              control = list(seed = 12345),
                                              verbose = F)
  })
  
  metricsView_scaled = metricsView
  for (i in 2:ncol(metricsView_scaled)){
    minVal = min(metricsView_scaled[[i]])
    maxDiff = max(metricsView_scaled[[i]]) - minVal
    metricsView_scaled[[i]] = (metricsView_scaled[[i]]-minVal)/maxDiff
  }
  
  g2 = ggplot(metricsView_scaled %>% gather(key = "metric", "value", -topics) %>% mutate(approach = ifelse(metric == "Griffiths2004", "Maximize", ".Minimize"))) +
    geom_line(aes(x = topics, y = value, color = metric)) +
    facet_wrap(vars(approach), ncol = 1) +
    scale_x_continuous(n.breaks = 10)+
    labs(x = "k")
  #ldatuning::FindTopicsNumber_plot(reviewOpt) # above does the same as this, but needed to return the chart object to create the grob
  
  #3
  perplexityDf = data.frame(train = numeric(), test = numeric())
  burnin = 100
  iter = 220
  keep = 50
  set.seed(12345)
  
  for (i in ks){
    fitted = LDA(dtm_train, k = i, method = "Gibbs",
                 control = list(burnin = burnin, iter = iter, keep = keep))
    perplexityDf[i,1] = topicmodels::perplexity(fitted, newdata = dtm_train)
    perplexityDf[i,2] = topicmodels::perplexity(fitted, newdata = dtm_test)
  }
  
  g3 = ggplot(perplexityDf %>% mutate(Topics = row_number()) %>% gather(key = "set", value = "Perplexity", train, test), aes(x = as.numeric(row.names(perplexityDf)))) +
    geom_line(aes(x = Topics, y = Perplexity, color = set)) +
    theme(axis.text.y = element_blank())+
    labs(x = "k")
  #return(gridExtra::grid.arrange(g2, g1, g3, widths = c(7,9,1.5), layout_matrix = rbind(c(1,1,1),c(2,3,NA))))
  return(list(g2, g1, g3))
}

plotKs = function(KCharts){
  gridExtra::grid.arrange(KCharts[[1]], KCharts[[2]], KCharts[[3]], widths = c(7, 9, 1.5), layout_matrix = rbind(c(1,1,1),c(2,3,NA)))
}
