library(tidyverse)
jhu <- readRDS("analysis/data/data-derived/jhu_all.rds")

# figure saving
save_figs <- function(name,
                      fig,
                      width = 6,
                      height = 6,
                      plot_dir = file.path(here::here(), "analysis/plots"),
                      pdf_plot = TRUE,
                      font_family = "Helvetica",
                      res = 300,
                      ...) {

  if(!is.null(font_family)) {
    fig <- fig + ggplot2::theme(text = ggplot2::element_text(family = font_family))
  }

  dir.create(plot_dir, showWarnings = FALSE)
  fig_path <- function(name) {paste0(plot_dir, "/", name)}

  ragg::agg_png(fig_path(paste0(name,".png")),
                width = width,
                height = height,
                units = "in",
                res = res,
                ...)
  print(fig)
  dev.off()

  if(pdf_plot) {
    pdf(file = fig_path(paste0(name,".pdf")), width = width, height = height)
    print(fig)
    dev.off()
  }

}

# cases plot
cases_plot <- function(data) {

  # Plot
  gg_cases <- ggplot2::ggplot() +
    ggplot2::geom_point(data = data,
                      mapping = ggplot2::aes(x = .data$date, y = .data$cases),
                      stat = "identity",
                      show.legend = TRUE,
                      inherit.aes = FALSE) +
    ggplot2::geom_line(data = data,
                      mapping = ggplot2::aes(x = .data$date, y = zoo::rollmean(.data$cases,na.pad = TRUE, k = 7)),
                      show.legend = TRUE, color = "red", lwd = 1,
                      inherit.aes = FALSE) +
    ggplot2::ylab("Daily Number of COVID-19 Infections") +
    ggplot2::theme_bw()  +
    ggplot2::scale_y_continuous(expand = c(0,0)) +
    ggplot2::scale_x_date(date_breaks = "2 months", date_labels = "%b %Y") +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, colour = "black"),
                   axis.title.x = ggplot2::element_blank(),
                   panel.grid.major.x = ggplot2::element_blank(),
                   panel.grid.minor.x = ggplot2::element_blank(),
                   panel.border = ggplot2::element_blank(),
                   panel.background = ggplot2::element_blank(),
                   axis.line = ggplot2::element_line(colour = "black")
    )

  gg_cases + ggplot2::theme(legend.position = "top",
                            legend.justification = c(0,1),
                            legend.direction = "horizontal", ) +
    ggplot2::ggtitle(paste0(data$Region[1], " COVID-19 Epidemic"))

}

# set up the results directory
isos <- unique(jhu$countryterritoryCode)
dir.create("analysis/plots/country_plots/", recursive = TRUE)

# make the plots
for(i in seq_along(isos)) {

  save_figs(paste0(isos[i]),
         cases_plot(jhu %>% filter(countryterritoryCode == isos[i])),
         width = 8,
         height = 6,
         res = 300, plot_dir = "analysis/plots/country_plots/",
         font_family = "Helvetica"
          )

}
