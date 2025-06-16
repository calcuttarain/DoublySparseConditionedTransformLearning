function plot_magnitude(train_data, test_data, kappas, T0, filename)
    for i = 1:3
        train_data{i} = compute_mean_sorted_vector(train_data{i});
        test_data{i}  = compute_mean_sorted_vector(test_data{i});
    end

    figure('Position', [100, 100, 1200, 600]);

    for i = 1:3
        subplot(2, 3, i);
        plot_sorted_stem(train_data{i}, T0);
        title(sprintf('$\\kappa(W) = %.2f$ (training)', kappas(i)), 'Interpreter', 'latex');

        subplot(2, 3, i + 3);
        plot_sorted_stem(test_data{i}, T0);
        title(sprintf('$\\kappa(W) = %.2f$ (test)', kappas(i)), 'Interpreter', 'latex');
    end

    if ~exist('plots', 'dir')
        mkdir('plots');
    end
    exportgraphics(gcf, fullfile('plots', filename), 'Resolution', 300);



    function mean_sorted_vector = compute_mean_sorted_vector(matrix)
        abs_matrix = abs(matrix);
        sorted_matrix = sort(abs_matrix, 1, 'descend');
        mean_sorted_vector = mean(sorted_matrix, 2);
    end

    function plot_sorted_stem(v, T0)
        x = 1:length(v);
        stem(x(1:T0), v(1:T0), 'Color', 'r', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
        hold on;
        stem(x(T0+1:end), v(T0+1:end), 'Color', [0.5 0.5 0.5], ...
             'MarkerFaceColor', [0.5 0.5 0.5], 'LineWidth', 1.5);
        hold off;
        xlabel('Transform Domain Components');
        ylabel('Magnitude');
    end
end
