// app/static/checkout/checkout.js
// Cygnus Payments — checkout client (world-class UX)
(function () {
  "use strict";

  const token = document.documentElement.dataset.checkoutToken;
  const apiBase = `/api/checkout/${token}`;

  // Core elements
  const statusBanner = document.getElementById("status-banner");
  const statusText =
    statusBanner.querySelector(".status-banner__text") || statusBanner;
  const addressSection = document.getElementById("address-section");
  const doneSection = document.getElementById("done-section");

  // Fields
  const areaEl = document.getElementById("area");
  const postalCodeEl = document.getElementById("postal_code");
  const houseNumberEl = document.getElementById("house_number");
  const streetEl = document.getElementById("street");
  const secondaryAddressEl = document.getElementById("secondary_address");

  // Error slots
  const houseNumberErrorEl = document.getElementById("house_number_error");
  const streetErrorEl = document.getElementById("street_error");

  // Feedback / actions
  const locationResult = document.getElementById("location-result");
  const locationResultText = document.getElementById("location-result-text");
  const saveSuccess = document.getElementById("save-success");

  const saveAddressBtn = document.getElementById("save-address-btn");
  const locationBtn = document.getElementById("location-btn");
  const continuePaymentBtn = document.getElementById("continue-payment-btn");
  const doneMessageEl = document.getElementById("done-message");
  const paymentModal = document.getElementById("payment-modal");
  const paymentIframe = document.getElementById("payment-iframe");
  const paymentModalCloseBtn = document.getElementById("payment-modal-close-btn");
  const paymentModalOpenTabBtn = document.getElementById("payment-modal-open-tab-btn");

  // Footer year
  const yrEl = document.getElementById("yr");
  if (yrEl) yrEl.textContent = String(new Date().getFullYear());

  let currentSession = null;
  let _paymentPollTimer = null;   // setInterval handle for frontend polling
  let currentPaymentUrl = "";
  let currentEmbeddedPaymentUrl = "";
  let currentPaymentPopup = null;

  // ----- Helpers -----
  function setStatus(message, variant = "info") {
    statusText.textContent = message;
    statusBanner.className = `status-banner status-banner--${variant}`;
  }

  function setContinueEnabled(enabled) {
    continuePaymentBtn.disabled = !enabled;
  }

  function rememberPaymentTargets(source) {
    if (!source) return;
    currentPaymentUrl =
      source.payment_link_url ||
      source.redirect_url ||
      currentPaymentUrl;
    currentEmbeddedPaymentUrl =
      source.payment_link_embedded_url ||
      source.embedded_url ||
      currentEmbeddedPaymentUrl;
  }

  function closePaymentModal() {
    if (!paymentModal) return;
    paymentModal.classList.add("hidden");
    paymentModal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-open");
    if (paymentIframe) {
      paymentIframe.src = "about:blank";
    }
  }

  function closePaymentPopup() {
    if (!currentPaymentPopup) return;
    try {
      if (!currentPaymentPopup.closed) {
        currentPaymentPopup.close();
      }
    } catch (_) {
      // Best-effort close for script-opened cross-origin windows.
    }
    currentPaymentPopup = null;
  }

  function closePaymentSurface() {
    closePaymentModal();
    closePaymentPopup();
  }

  function showPaymentModal(url) {
    if (!paymentModal || !paymentIframe || !url) return;
    paymentIframe.src = url;
    paymentModal.classList.remove("hidden");
    paymentModal.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
  }

  function openPaymentSurface() {
    if (currentEmbeddedPaymentUrl) {
      showPaymentModal(currentEmbeddedPaymentUrl);
      return;
    }

    if (currentPaymentUrl) {
      currentPaymentPopup = window.open(
        currentPaymentUrl,
        "compass-payment",
        "popup=yes,width=540,height=820,menubar=no,toolbar=no,location=yes,status=no,resizable=yes,scrollbars=yes"
      );

      if (!currentPaymentPopup) {
        window.open(currentPaymentUrl, "_blank", "noopener,noreferrer");
      }
    }
  }

  function setLoading(btn, loading) {
    if (!btn) return;
    if (loading) {
      btn.dataset._origDisabled = btn.disabled ? "1" : "0";
      btn.classList.add("is-loading");
      btn.disabled = true;
    } else {
      btn.classList.remove("is-loading");
      btn.disabled = btn.dataset._origDisabled === "1";
      delete btn.dataset._origDisabled;
    }
  }

  function setFieldError(inputEl, errorEl, message) {
    const field = inputEl.closest(".field");
    if (!field || !errorEl) return;

    if (message) {
      field.classList.add("field--invalid");
      errorEl.textContent = message;
    } else {
      field.classList.remove("field--invalid");
      errorEl.textContent = "";
    }
  }

  function clearValidation() {
    setFieldError(houseNumberEl, houseNumberErrorEl, "");
    setFieldError(streetEl, streetErrorEl, "");
  }

  function showSaveSuccess(show) {
    saveSuccess.classList.toggle("hidden", !show);
  }

  function setLocationSummary(session) {
    const parts = [
      session.house_number,
      session.street,
      session.area,
      session.postal_code,
      session.city,
      session.state,
    ].filter(Boolean);

    if (session.address_source === "device_location" && parts.length) {
      locationResult.classList.remove("hidden");
      locationResultText.textContent = parts.join(", ");
      return;
    }

    locationResult.classList.add("hidden");
    locationResultText.textContent = "";
  }

  function populateForm(session) {
    areaEl.value = session.area || "";
    postalCodeEl.value = session.postal_code || "";
    houseNumberEl.value = session.house_number || "";
    streetEl.value = session.street || "";
    secondaryAddressEl.value = session.secondary_address || "";
    setLocationSummary(session);
  }

  /**
   * Lock the entire address form so nothing can be re-submitted
   * once the address (or payment) is already completed.
   */
  function lockAddressForm() {
    // Disable editable inputs
    [houseNumberEl, streetEl, secondaryAddressEl].forEach((el) => {
      el.disabled = true;
      el.closest(".field").classList.add("field--locked");
    });

    // Disable both action buttons
    saveAddressBtn.disabled = true;
    saveAddressBtn.classList.add("btn--locked");
    locationBtn.disabled = true;
    locationBtn.classList.add("btn--locked");

    // Mark the address section as locked for CSS styling
    const addressMain = document.querySelector(".checkout-main");
    if (addressMain) addressMain.classList.add("form--locked");
  }

  /**
   * Unlock the form (used on initial load when nothing is completed yet).
   */
  function unlockAddressForm() {
    [houseNumberEl, streetEl, secondaryAddressEl].forEach((el) => {
      el.disabled = false;
      el.closest(".field").classList.remove("field--locked");
    });
    saveAddressBtn.disabled = false;
    saveAddressBtn.classList.remove("btn--locked");
    locationBtn.disabled = false;
    locationBtn.classList.remove("btn--locked");

    const addressMain = document.querySelector(".checkout-main");
    if (addressMain) addressMain.classList.remove("form--locked");
  }

  function advanceStepper(step) {
    // step: 1 = address, 2 = payment, 3 = done
    const steps = document.querySelectorAll(".checkout-step");
    steps.forEach((el, idx) => {
      el.classList.toggle("checkout-step--active", idx + 1 <= step);
    });
  }

  function renderSession(session) {
    currentSession = session;
    rememberPaymentTargets(session);

    // Always start with known state
    addressSection.classList.remove("hidden");
    if (paymentSection) paymentSection.classList.add("hidden");
    doneSection.classList.add("hidden");
    stopPaymentPolling();
    closePaymentSurface();

    populateForm(session);

    // ── Payment already completed ──────────────────────────────────────────
    if (session.payment_completed) {
      lockAddressForm();
      addressSection.classList.add("hidden");
      if (paymentSection) paymentSection.classList.add("hidden");
      doneSection.classList.remove("hidden");
      const orderNumber = session.order_number || "";
      const paymentRef  = session.payment_reference || "";
      doneMessageEl.textContent =
        `Order #${orderNumber} is confirmed. Payment reference: ${paymentRef}`.trim();
      advanceStepper(3);
      setStatus(`Payment confirmed for Order #${orderNumber}.`, "success");
      if (paymentSpinner) paymentSpinner.classList.add("hidden");
      return;
    }

    // ── Payment started but not complete → resume waiting screen ──────────
    // This handles the case where the user refreshes the page mid-payment.
    if (session.payment_started) {
      lockAddressForm();
      addressSection.classList.add("hidden");
      if (paymentSection) paymentSection.classList.remove("hidden");
      advanceStepper(2);
      setStatus("Waiting for payment confirmation…", "info");
      if (paymentStatusText) {
        paymentStatusText.textContent = currentEmbeddedPaymentUrl
          ? "Your secure payment window is ready inside this page. If you closed it, reopen it below. This page will confirm your order automatically."
          : "We're waiting for payment confirmation. You can reopen the payment page if needed, and this page will confirm your order automatically.";
      }
      if (paymentSpinner) paymentSpinner.classList.remove("hidden");
      if (paymentOpenBtn) {
        paymentOpenBtn.onclick = () => openPaymentSurface();
        paymentOpenBtn.disabled = !(currentEmbeddedPaymentUrl || currentPaymentUrl);
      }
      // Resume polling — the backend is already polling Datacap; frontend polls
      // too so the page auto-updates without any user interaction.
      startPaymentPolling();
      return;
    }

    // ── Address already saved → lock the form, enable Continue ────────────
    if (session.address_completed) {
      lockAddressForm();
      setContinueEnabled(true);
      showSaveSuccess(true);
      advanceStepper(2);

      if (session.address_source === "device_location") {
        setStatus(
          "Current location saved. Your delivery address is ready for payment.",
          "success"
        );
      } else {
        setStatus(
          "Delivery address saved. You can continue to secure payment.",
          "success"
        );
      }
      return;
    }

    // ── Fresh / incomplete ─────────────────────────────────────────────────
    unlockAddressForm();
    showSaveSuccess(false);
    setContinueEnabled(false);
    advanceStepper(1);
    setStatus("Enter your delivery address to continue.", "info");
  }

  // ----- Payment-waiting section helpers -----
  const paymentSection = document.getElementById("payment-section");
  const paymentOpenBtn = document.getElementById("payment-open-btn");
  const paymentStatusText = document.getElementById("payment-status-text");
  const paymentSpinner = document.getElementById("payment-spinner");

  function showPaymentWaiting(paymentUrl, embeddedUrl) {
    rememberPaymentTargets({
      redirect_url: paymentUrl,
      embedded_url: embeddedUrl,
    });

    // Hide address section, show payment waiting section
    addressSection.classList.add("hidden");
    if (paymentSection) paymentSection.classList.remove("hidden");
    doneSection.classList.add("hidden");
    advanceStepper(2);
    setStatus("Waiting for payment confirmation…", "info");

    // Wire up "Open Payment Page" button
    if (paymentOpenBtn) {
      paymentOpenBtn.onclick = () => openPaymentSurface();
      paymentOpenBtn.disabled = false;
    }
    if (paymentStatusText) {
      paymentStatusText.textContent = currentEmbeddedPaymentUrl
        ? "Your secure payment page is open inside this page. Complete payment there, and this page will confirm your order automatically."
        : "Your secure payment page has been opened. Complete your payment there, and this page will update automatically.";
    }
    if (paymentSpinner) paymentSpinner.classList.remove("hidden");

    openPaymentSurface();

    // Start polling backend every 5s
    startPaymentPolling();
  }

  function startPaymentPolling() {
    stopPaymentPolling(); // clear any existing
    _paymentPollTimer = setInterval(async () => {
      try {
        const res = await fetch(`${apiBase}/verify-payment`, { method: "POST" });
        if (!res.ok) return; // transient error — keep polling
        const result = await res.json();

        if (result.payment_completed || result.paid) {
          stopPaymentPolling();
          // Render the completed state using the session returned by the server
          if (result.session) {
            renderSession(result.session);
          } else {
            // Reload session from server
            await loadSession();
          }
        } else if (result.status && ["cancelled", "canceled", "expired", "failed", "declined"].includes(result.status.toLowerCase())) {
          stopPaymentPolling();
          closePaymentSurface();
          setStatus("Payment was not completed. Please try again.", "error");
          if (paymentStatusText) paymentStatusText.textContent = "Payment failed or was cancelled. Please go back and try again.";
          if (paymentSpinner) paymentSpinner.classList.add("hidden");
        }
        // else: still open/pending — keep polling
      } catch (_) {
        // Network error — keep polling silently
      }
    }, 5000);
  }

  function stopPaymentPolling() {
    if (_paymentPollTimer !== null) {
      clearInterval(_paymentPollTimer);
      _paymentPollTimer = null;
    }
  }

  // ----- API calls -----
  async function loadSession() {
    const res = await fetch(apiBase);
    if (!res.ok) {
      throw new Error("Failed to load checkout session.");
    }
    const data = await res.json();
    renderSession(data);
  }

  function validateAddress() {
    clearValidation();

    const houseNumber = houseNumberEl.value.trim();
    const street = streetEl.value.trim();
    let firstInvalid = null;
    let hasError = false;

    if (!houseNumber) {
      setFieldError(houseNumberEl, houseNumberErrorEl, "House number is required.");
      firstInvalid = firstInvalid || houseNumberEl;
      hasError = true;
    }

    if (!street) {
      setFieldError(streetEl, streetErrorEl, "Street is required.");
      firstInvalid = firstInvalid || streetEl;
      hasError = true;
    }

    if (hasError) {
      if (firstInvalid && typeof firstInvalid.focus === "function") {
        firstInvalid.focus();
      }
      throw new Error("Please correct the highlighted address fields.");
    }

    return {
      house_number: houseNumber,
      street: street,
      secondary_address: secondaryAddressEl.value.trim() || null,
    };
  }

  async function saveAddress() {
    const payload = validateAddress();

    const res = await fetch(`${apiBase}/address`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(body.detail || "Failed to save address.");
    }

    renderSession(body);
  }

  async function saveLocationPayload(payload) {
    const res = await fetch(`${apiBase}/location`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(body.detail || "Could not save location.");
    }

    return body;
  }

  async function useCurrentLocation() {
    if (!navigator.geolocation) {
      throw new Error("Geolocation is not supported on this device.");
    }

    setStatus("Requesting your current location…", "info");
    showSaveSuccess(false);

    return new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          try {
            const payload = {
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
              permission_granted: true,
            };

            const updated = await saveLocationPayload(payload);
            renderSession(updated);
            resolve(updated);
          } catch (error) {
            reject(error);
          }
        },
        async () => {
          try {
            await saveLocationPayload({
              latitude: null,
              longitude: null,
              permission_granted: false,
            });
            reject(new Error("Location access was not granted."));
          } catch (error) {
            reject(error);
          }
        },
        {
          enableHighAccuracy: true,
          timeout: 12000,
          maximumAge: 0,
        }
      );
    });
  }

  async function goToPayment() {
    try {
      setStatus("Initializing secure payment…", "info");
      setLoading(continuePaymentBtn, true);

      const summary = (currentSession && currentSession.order_summary) || {};
      const amount =
        summary.total ||
        summary.total_price ||
        summary.grand_total ||
        "0.00";

      const res = await fetch(`${apiBase}/payment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ amount: String(amount) }),
      });

      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(body.detail || "Unable to initialize payment.");
      }

      if (body.redirect_url || body.embedded_url) {
        // Instead of navigating away (which loses our polling page), we open the
        // payment surface in-page when Datacap provides an embedded URL. If not,
        // we fall back to a new tab and keep this confirmation page alive.
        // The backend is already polling Datacap every 6 s; the frontend joins
        // with its own 5 s poll so the page auto-updates the moment payment clears.
        showPaymentWaiting(body.redirect_url, body.embedded_url);
        setLoading(continuePaymentBtn, false);
        return;
      }

      throw new Error("Payment URL was not returned.");
    } catch (error) {
      setStatus(error.message || "Unable to proceed to payment.", "error");
      setLoading(continuePaymentBtn, false);
      continuePaymentBtn.disabled = false;
    }
  }

  // ----- Event wiring -----
  houseNumberEl.addEventListener("input", () => {
    if (houseNumberEl.value.trim()) {
      setFieldError(houseNumberEl, houseNumberErrorEl, "");
    }
    showSaveSuccess(false);
  });

  streetEl.addEventListener("input", () => {
    if (streetEl.value.trim()) {
      setFieldError(streetEl, streetErrorEl, "");
    }
    showSaveSuccess(false);
  });

  secondaryAddressEl.addEventListener("input", () => {
    showSaveSuccess(false);
  });

  // Enter-to-save in any address field
  [houseNumberEl, streetEl, secondaryAddressEl].forEach((el) => {
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        saveAddressBtn.click();
      }
    });
  });

  saveAddressBtn.addEventListener("click", async () => {
    try {
      setLoading(saveAddressBtn, true);
      setStatus("Saving address…", "info");
      await saveAddress();
    } catch (error) {
      setStatus(error.message || "Failed to save address.", "error");
    } finally {
      setLoading(saveAddressBtn, false);
    }
  });

  locationBtn.addEventListener("click", async () => {
    try {
      setLoading(locationBtn, true);
      await useCurrentLocation();
    } catch (error) {
      setStatus(error.message || "Failed to use current location.", "error");
    } finally {
      setLoading(locationBtn, false);
    }
  });

  continuePaymentBtn.addEventListener("click", async () => {
    await goToPayment();
  });

  if (paymentModalCloseBtn) {
    paymentModalCloseBtn.addEventListener("click", () => {
      closePaymentSurface();
    });
  }

  if (paymentModalOpenTabBtn) {
    paymentModalOpenTabBtn.addEventListener("click", () => {
      if (currentPaymentUrl || currentEmbeddedPaymentUrl) {
        window.open(
          currentPaymentUrl || currentEmbeddedPaymentUrl,
          "_blank",
          "noopener,noreferrer"
        );
      }
    });
  }

  document.querySelectorAll("[data-payment-modal-close]").forEach((el) => {
    el.addEventListener("click", () => {
      closePaymentSurface();
    });
  });

  // Initial load
  loadSession().catch((error) => {
    setStatus(error.message || "Failed to load checkout.", "error");
  });
})();
