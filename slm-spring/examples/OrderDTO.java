package com.example.api.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import java.util.UUID;

/**
 * Order DTO with nested line-item validation. The @Valid annotation on the
 * collection cascades validation into each OrderLineDTO.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OrderDTO {

    public enum Status { PENDING, PAID, SHIPPED, DELIVERED, CANCELLED }

    private UUID id;

    @NotNull(message = "customerId is required")
    private UUID customerId;

    @NotNull(message = "status is required")
    private Status status;

    @NotEmpty(message = "order must have at least one line item")
    @Valid
    private List<OrderLineDTO> lines;

    @NotNull
    @DecimalMin(value = "0.00", message = "total cannot be negative")
    private BigDecimal total;

    private Instant createdAt;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class OrderLineDTO {

        @NotNull(message = "productId is required")
        private Long productId;

        @NotNull(message = "quantity is required")
        @Positive(message = "quantity must be positive")
        private Integer quantity;

        @NotNull(message = "unitPrice is required")
        @DecimalMin(value = "0.00", message = "unitPrice cannot be negative")
        private BigDecimal unitPrice;
    }
}
