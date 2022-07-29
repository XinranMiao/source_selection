library(tidyverse)

expand.grid(country = c("France", "Latvija", "United Kingdom", "Sverige", "Slovensko"),
            algorithm = c("bandit", "random"),
            target_size = c(320, 640, 1600)) %>%
  write.table(file = "args.txt", quote = FALSE, sep = ",",
              col.names = FALSE, row.names = FALSE)
