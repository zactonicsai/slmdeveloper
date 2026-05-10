package com.example.api.controller;

import com.example.api.dto.OrderDTO;
import com.example.api.service.OrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;
import java.util.List;
import java.util.UUID;

/**
 * REST controller for Orders. Status transitions go through PATCH; full updates
 * are intentionally omitted because orders are append-only after creation.
 */
@RestController
@RequestMapping("/api/v1/orders")
@RequiredArgsConstructor
@Slf4j
public class OrderController {

    private final OrderService orderService;

    @GetMapping
    public ResponseEntity<List<OrderDTO>> listOrders(
            @RequestParam(required = false) UUID customerId,
            @RequestParam(required = false) OrderDTO.Status status) {
        log.debug("GET /api/v1/orders customerId={} status={}", customerId, status);
        return ResponseEntity.ok(orderService.find(customerId, status));
    }

    @GetMapping("/{id}")
    public ResponseEntity<OrderDTO> getOrder(@PathVariable UUID id) {
        return ResponseEntity.ok(orderService.findById(id));
    }

    @PostMapping
    public ResponseEntity<OrderDTO> createOrder(
            @Valid @RequestBody OrderDTO request,
            UriComponentsBuilder uriBuilder) {
        OrderDTO created = orderService.create(request);
        URI location = uriBuilder.path("/api/v1/orders/{id}")
                .buildAndExpand(created.getId())
                .toUri();
        return ResponseEntity.created(location).body(created);
    }

    @PatchMapping("/{id}/status")
    public ResponseEntity<OrderDTO> updateStatus(
            @PathVariable UUID id,
            @RequestParam OrderDTO.Status status) {
        log.info("PATCH /api/v1/orders/{}/status -> {}", id, status);
        return ResponseEntity.ok(orderService.updateStatus(id, status));
    }
}
