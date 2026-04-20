log_file <- "detection_log.csv"
event_log_file <- "event_log.csv"
unlock_confidence_threshold <- 90.0
unlock_duration_seconds <- 5.0
max_failed_attempts <- 5
lockdown_duration_seconds <- 15
anomaly_frequency_per_min <- 600
anomaly_confidence_avg <- 60.0

lockdown_active <- FALSE
lockdown_end_time <- Sys.time()
failed_attempts <- 0
last_processed_line <- 0
last_unlock_time <- as.POSIXct("1970-01-01")
last_failure_increment_time <- as.POSIXct("1970-01-01")
failure_persistence_start <- NULL

parse_time <- function(t_str) {
    as.POSIXct(t_str, format = "%Y-%m-%d %H:%M:%S")
    as.POSIXct(t_str, format = "%Y-%m-%d %H:%M:%S")
}

log_event <- function(type, message) {
    timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    if (!file.exists(event_log_file)) {
        writeLines("Timestamp,EventType,Message", event_log_file)
    }
    line <- sprintf("%s,%s,\"%s\"", timestamp, type, message)
    # Using a file connection with close() to force immediate disk write
    con <- file(event_log_file, "a")
    writeLines(line, con)
    close(con)
}

check_anomalies <- function(recent_data) {
    if (nrow(recent_data) == 0) {
        return()
    }

    now <- Sys.time()
    recent_1min <- recent_data[difftime(now, recent_data$Timestamp,
        units = "secs"
    ) <= 60, ]
    if (nrow(recent_1min) > anomaly_frequency_per_min) {
        msg <- sprintf("[ANOMALY] High Frequency: %d/min", nrow(recent_1min))
        cat(paste0(msg, "\n"))
        writeLines(msg, "supervisory_msg.txt")
        log_event("ANOMALY_FREQ", msg)
    }

    avg_conf <- mean(recent_data$Confidence)
    if (avg_conf < anomaly_confidence_avg) {
        cat(sprintf(
            "[ANOMALY] Low Avg Confidence (%.1f%%). Potential Spoofing.\n",
            avg_conf
        ))
        log_event("ANOMALY_CONF", sprintf("Low Avg Confidence (%.1f%%)", avg_conf))
    }

    hour <- as.numeric(format(now, "%H"))
    if (hour >= 21 || hour < 9) {
        cat(sprintf(
            "[ANOMALY] Access attempt at unusual time: %s\n",
            format(now, "%H:%M")
        ))
        log_event("ANOMALY_TIME", sprintf("Unusual time: %s", format(now, "%H:%M")))
    }

    unique_people <- length(unique(recent_data$Label))
    if (unique_people > 2) {
        cat(sprintf(
            "[ANOMALY] Multiple Identities (%d). Possible intrusion.\n",
            unique_people
        ))
        log_event("ANOMALY_IDENTITY", sprintf("Multiple Identities (%d)", unique_people))
    }
}

update_dashboard <- function(data) {
    if (nrow(data) < 5) {
        return()
    }

    # Create a subset for time-series plots to keep them readable (last 5 mins or 1000 points)
    recent_data <- data
    if (nrow(data) > 1000) {
        now <- Sys.time()
        recent_data <- data[difftime(now, data$Timestamp, units = "mins") <= 5, ]
    }

    if (nrow(recent_data) == 0) recent_data <- data # Fallback if filtering removes everything

    # Open device if not open
    if (is.null(dev.list())) {
        if (.Platform$OS.type == "windows") {
            try(windows(width = 12, height = 10, title = "Supervisory Dashboard"))
        } else {
            try(x11(width = 12, height = 10, title = "Supervisory Dashboard"))
        }
    }

    # Set layout: 2x2 grid
    par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))

    # --- Plot 1: Confidence Trends (Time Series) ---
    labels <- unique(recent_data$Label)
    colors <- rainbow(length(labels))
    label_colors <- colors[match(recent_data$Label, labels)]

    plot(recent_data$Timestamp, recent_data$Confidence,
        main = "Confidence Trends (Recent)",
        xlab = "Time", ylab = "Confidence (%)",
        col = label_colors, pch = 19, cex = 0.6,
        ylim = c(0, 100)
    )
    abline(h = 90, col = "red", lty = 2, lwd = 1)
    grid()
    legend("bottomleft", legend = labels, col = colors, pch = 19, cex = 0.8, bg = "white", inset = 0.02)

    # --- Plot 2: Anomaly Rate (Time Series) ---
    is_anomaly <- (recent_data$Label == "Unknown" | recent_data$Confidence < 80)
    window_size <- 20
    if (length(is_anomaly) >= window_size) {
        anomaly_rate <- stats::filter(as.numeric(is_anomaly), rep(1 / window_size, window_size), sides = 1)
        plot(recent_data$Timestamp, anomaly_rate * 100,
            type = "l", lwd = 2, col = "darkorange",
            main = "Anomaly Rate (Rolling)",
            xlab = "Time", ylab = "Rate (%)",
            ylim = c(0, 100)
        )
        abline(h = 50, col = "red", lty = 3)
        grid()
    } else {
        plot.new()
        text(0.5, 0.5, "Not enough data for Rate")
    }

    # --- Plot 3: Detected Faces Distribution (Pie Chart) ---
    # Use full 'data' for distribution to show overall session stats
    face_counts <- table(data$Label)
    if (length(face_counts) > 0) {
        pct <- round(100 * face_counts / sum(face_counts))
        lbls <- paste(names(face_counts), "\n", pct, "%", sep = "")
        pie(face_counts,
            labels = lbls, main = "Detected Faces Distribution",
            col = rainbow(length(face_counts))
        )
    } else {
        plot.new()
        text(0.5, 0.5, "No Faces Detected")
    }

    # --- Plot 4: Anomaly Types Distribution (Bar Chart) ---
    if (file.exists("event_log.csv")) {
        events <- tryCatch(read.csv("event_log.csv", stringsAsFactors = FALSE), error = function(e) NULL)
        if (!is.null(events) && nrow(events) > 0) {
            # Filter for Anomaly events
            anomalies <- events[grep("ANOMALY", events$EventType), ]
            print(paste("Anomalies found:", nrow(anomalies))) # Debug

            if (nrow(anomalies) > 0) {
                type_counts <- table(anomalies$EventType)
                # Shorten labels for cleaner bar chart
                names(type_counts) <- gsub("ANOMALY_", "", names(type_counts))

                barplot(type_counts,
                    main = "Anomaly Distribution by Type",
                    col = "tomato", border = "darkred",
                    las = 2, cex.names = 0.7
                ) # las=2 rotates x-labels
            } else {
                plot.new()
                text(0.5, 0.5, "No Anomalies in Log")
            }
        } else {
            plot.new()
            text(0.5, 0.5, "Event Log Empty")
        }
    } else {
        plot.new()
        text(0.5, 0.5, "No Event Log File")
    }
}

monitor_loop <- function() {
    cat("Starting Supervisory System...\n")
    cat(sprintf("Waiting for data in %s...\n", log_file))

    while (TRUE) {
        if (!file.exists(log_file)) {
            Sys.sleep(1)
            next
        }

        data <- tryCatch(read.csv(log_file, stringsAsFactors = FALSE),
            error = function(e) NULL
        )
        if (is.null(data) || nrow(data) == 0) {
            Sys.sleep(0.5)
            next
        }

        if (nrow(data) <= last_processed_line) {
            Sys.sleep(0.1)
            next
        }

        data$Timestamp <- parse_time(data$Timestamp)
        data$Confidence <- as.numeric(data$Confidence)

        new_data <- data[(last_processed_line + 1):nrow(data), ]
        last_processed_line <<- nrow(data)

        if (lockdown_active) {
            remaining <- difftime(lockdown_end_time, Sys.time(), units = "secs")
            if (remaining > 0) {
                if (as.numeric(Sys.time()) %% 5 < 0.2) {
                    msg <- sprintf("[LOCKDOWN] Ignoring input for %.0f s...", remaining)
                    cat(paste0("\r", msg))
                    writeLines(msg, "supervisory_msg.txt")
                }
                Sys.sleep(0.1)
                next
            } else {
                cat("\n[SYSTEM] Lockdown Lifted.\n")
                lockdown_active <<- FALSE
                failed_attempts <<- 0
                failure_persistence_start <<- NULL
                writeLines("0", "lockdown_signal.txt")
                log_event("LOCKDOWN_END", "Lockdown Lifted")
            }
        }

        latest <- tail(new_data, 1)
        now <- latest$Timestamp
        failure_window_data <- data[difftime(now, data$Timestamp,
            units = "secs"
        ) <= 1.0, ]

        is_failure_frame <- (latest$Label == "Unknown" ||
            latest$Confidence < unlock_confidence_threshold)

        if (is_failure_frame) {
            if (nrow(failure_window_data) > 5) {
                bad_count <- sum(failure_window_data$Label == "Unknown" |
                    failure_window_data$Confidence < unlock_confidence_threshold)
                if (bad_count / nrow(failure_window_data) > 0.9) {
                    if (difftime(now, last_failure_increment_time,
                        units = "secs"
                    ) > 2.0) {
                        failed_attempts <<- failed_attempts + 1
                        last_failure_increment_time <<- now
                        msg <- sprintf("ACCESS DENIED - Attempt %d/%d", failed_attempts, max_failed_attempts)
                        cat(paste0(msg, "\n"))
                        writeLines(msg, "supervisory_msg.txt")
                        log_event("ACCESS_DENIED", msg)

                        if (failed_attempts >= max_failed_attempts) {
                            cat("[SECURITY] Max failed attempts reached! LOCKDOWN.\n")
                            lockdown_active <<- TRUE
                            lockdown_end_time <<- Sys.time() + lockdown_duration_seconds
                            writeLines("1", "lockdown_signal.txt")
                            log_event("LOCKDOWN_START", "Max failed attempts reached")
                        }
                    }
                }
            }
        } else {
            window_start <- now - unlock_duration_seconds
            valid_window_data <- data[
                data$Timestamp >= window_start & data$Timestamp <= now,
            ]

            if (nrow(valid_window_data) > 10) {
                all_same_user <- all(valid_window_data$Label == latest$Label)
                all_high_conf <- all(
                    valid_window_data$Confidence > unlock_confidence_threshold
                )

                if (all_same_user && all_high_conf) {
                    if (difftime(now, last_unlock_time, units = "secs") > 5.0) {
                        msg <- sprintf("ACCESS GRANTED: %s verified", latest$Label)
                        cat(paste0(msg, "\n"))
                        writeLines(msg, "supervisory_msg.txt")
                        log_event("ACCESS_GRANTED", sprintf("User %s verified", latest$Label))
                        last_unlock_time <<- now
                        failed_attempts <<- 0
                    }
                }
            }
        }

        if (runif(1) < 0.1) {
            check_anomalies(tail(data, 100))
        }

        # Check for stop signal
        if (file.exists("supervisory_stop.txt")) {
            cat("\n[SYSTEM] Stop signal received. Generating report...\n")

            # Wrap in try to prevent crash from stopping exit
            try(
                {
                    update_dashboard(data)
                },
                silent = FALSE
            )

            # Only wait if a graphics device was actually opened
            if (!is.null(dev.list())) {
                cat("[SYSTEM] Dashboard displayed. PLEASE CLOSE THE PLOT WINDOW TO EXIT.\n")
                while (!is.null(dev.list())) {
                    Sys.sleep(0.5)
                }
                cat("[SYSTEM] Window closed. Exiting.\n")
            } else {
                cat("[SYSTEM] No plot window detected or failed to open. Exiting.\n")
            }
            break
        }

        Sys.sleep(0.1)
    }
}

if (!interactive()) {
    monitor_loop()
}
