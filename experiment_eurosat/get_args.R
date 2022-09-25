library(tidyverse)

expand.grid(country = c("France", "Latvija", "United Kingdom", "Sverige", "Slovensko"),
            algorithm = c("bandit", "random"),
            target_size = c(320, 640, 1600)) %>%
  write.table(file = "args.txt", quote = FALSE, sep = ",",
              col.names = FALSE, row.names = FALSE)



# cluster the countries
expand.grid(target_task = 0:9,
            algorithm = c("bandit", "random"),
            target_size = c(160, 320, 640)) %>%
  write.table(file = "arg_clusters.txt", quote = FALSE, sep = ",",
              col.names = FALSE, row.names = FALSE)
